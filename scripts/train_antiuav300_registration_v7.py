"""Matched-budget train-only shared-velocity pilot, with optional MIND anchor.

No validation/test access; successful execution is NOT registration qualification.
Epoch-boundary resume is supported. A smoke run cannot masquerade as a full pilot.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.protocol_v2 import rotating_positions
from aero_ir.registration.protocol_v7 import (
    ARCHITECTURE,
    INTEGRATION_STEPS,
    LEARNING_RATE,
    LOSS_WEIGHTS,
    MIND_WEIGHT,
    SharedVelocityMatcher,
    observation,
    step,
)
from aero_ir.registration.qualification_v4 import direction_statistics, split_report
from aero_ir.registration.superfusion import DenseMatcher, load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256

INITIAL_SHA256 = "b719c61b6ba991a1855c66ca32d7cbe1d86646498a289c5fc2d434ba3cf6f02e"
ARRAY_KEYS = ("visible", "infrared", "source_boxes", "target_boxes")


def automatic_exclusion(image):
    """Same header/thin-reticle heuristic as the ROI diagnostic, plus8px guard."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    central = np.zeros_like(gray)
    for extreme in ((gray <= 10), (gray >= 245)):
        for horizontal in (True, False):
            length = max(16, round((w if horizontal else h) * 0.07))
            kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
            opened = cv2.morphologyEx(extreme.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
            for i in range(1, count):
                x, y, bw, bh, _ = stats[i]
                thickness, extent = (bh, bw) if horizontal else (bw, bh)
                if (
                    thickness <= 4
                    and extent >= length
                    and 0.18 * w <= x + bw / 2 <= 0.82 * w
                    and 0.18 * h <= y + bh / 2 <= 0.82 * h
                ):
                    central[labels == i] = 1
    mask = cv2.dilate(central, np.ones((3, 3), np.uint8))
    mask[: int(np.ceil(0.2 * h))] = 1
    return cv2.dilate(mask, np.ones((17, 17), np.uint8)).astype(bool)


def epoch_selection(shards, epoch, seed):
    selected = [
        (s["sequence_id"], int(i))
        for s in sorted(shards, key=lambda s: s["sequence_id"])
        for i in rotating_positions(s["pairs"], epoch, s["sequence_id"], seed=seed)
    ]
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(selected))
    return [selected[i] for i in order]


def read_batch(cache, selected):
    collected = {k: [] for k in ARRAY_KEYS}
    digest = hashlib.sha256()
    for sid, index in selected:
        folder = (cache / "shards" / sid).resolve()
        if not folder.is_relative_to((cache / "shards").resolve()):
            raise ValueError("sequence path outside cache")
        digest.update(json.dumps([sid, index]).encode())
        for k in ARRAY_KEYS:
            value = np.array(np.load(folder / f"{k}.npy", mmap_mode="r")[index])
            if not np.isfinite(value).all():
                raise ValueError("nonfinite cached array")
            digest.update(k.encode())
            digest.update(value.tobytes())
            collected[k].append(value)
    return {k: np.stack(v) for k, v in collected.items()}, digest.hexdigest()


def tensors(arrays, device):
    images = [
        torch.from_numpy(arrays[k]).permute(0, 3, 1, 2).to(device).float() / 255
        for k in ARRAY_KEYS[:2]
    ]
    boxes = [torch.from_numpy(arrays[k]).to(device) for k in ARRAY_KEYS[2:]]
    return (*images, *boxes)


def load_trained(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != ARCHITECTURE or "DM" in payload:
        raise ValueError("not a v7 shared-velocity checkpoint")
    if any(not torch.isfinite(v).all() for v in payload["shared_velocity_state"].values()):
        raise ValueError("nonfinite shared-velocity checkpoint")
    model = SharedVelocityMatcher(DenseMatcher()).to(device).eval()
    model.load_state_dict(payload["shared_velocity_state"], strict=True)
    return model, payload


def save_checkpoint(path, model, optimizer, scheduler, epoch, spec):
    temp = path.with_suffix(".partial")
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "shared_velocity_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_epochs": epoch,
            "spec_sha256": canonical_hash(spec),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generator_training_eligible": "hold_not_qualified",
        },
        temp,
    )
    os.replace(temp, path)


def screen(model, cache, shards, device):
    rows, samples = [], []
    # Predictor caches scale/grid tensors across calls. Inference-mode tensors
    # must not leak into the subsequent gradient-enabled training forward pass.
    with torch.no_grad():
        for shard in sorted(shards, key=lambda s: s["sequence_id"]):
            selected = [(shard["sequence_id"], shard["pairs"] // 2)]
            arrays, digest = read_batch(cache, selected)
            vis, ir, vb, ib = tensors(arrays, device)
            f, r = model.fields(ir, vis)
            rows.append(
                {
                    "sequence_id": shard["sequence_id"],
                    "ir_to_rgb_points": direction_statistics(f, r, ib, vb)[0],
                    "rgb_to_ir_points": direction_statistics(r, f, vb, ib)[0],
                }
            )
            samples.append({"selection": selected, "array_sha256": digest})
    return {
        "summary": split_report(rows),
        "rows": rows,
        "samples": samples,
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "not_independent_of_training": True,
        "generator_training_eligible": "hold_not_qualified",
    }


def write_json(path, obj):
    with path.open("x") as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300")
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path(
            "experiments/antiuav300_registration_v6_pilot_e10_seed0/antiuav300_registration_v6_e10.pth"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("geometry", "geometry_mind"), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or min(args.seed, args.smoke_steps) < 0:
        parser.error("invalid training budget")
    for key in ("root", "cache_root", "initial_checkpoint", "output_dir"):
        if any(c in str(getattr(args, key)) for c in ("\n", "\r")):
            parser.error(f"{key} contains a newline")
    if args.output_dir.exists() and not args.resume:
        parser.error("refusing overwrite; use a fresh directory or explicit --resume")
    if args.resume and (args.smoke_steps or not (args.output_dir / "latest.pth").is_file()):
        parser.error("resume requires a non-smoke run with an epoch-boundary checkpoint")
    if file_sha256(args.initial_checkpoint) != INITIAL_SHA256:
        parser.error("requires the completed frozen v6 seed0 checkpoint")
    cache_path = args.cache_root / "manifest.json"
    manifest = json.loads(cache_path.read_text())
    if (
        manifest.get("kind") != "antiuav300_registration_v2_full_train_cache"
        or manifest.get("fit_split") != "train"
        or manifest.get("validation_or_test_access") != "none"
        or canonical_hash({k: v for k, v in manifest.items() if k != "cache_manifest_sha256"})
        != manifest["cache_manifest_sha256"]
        or file_sha256(args.root / "label_new/train.json") != manifest["split_manifest_sha256"]
    ):
        parser.error("invalid or mismatching official train cache")
    initial_metadata = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)[
        "aero_registration"
    ]
    if initial_metadata.get("cache_manifest_sha256") != manifest["cache_manifest_sha256"]:
        parser.error("cache differs from the pinned initializer's training cache")
    shards = manifest["shards"]
    if (
        len(shards) != 160
        or len({s["sequence_id"] for s in shards}) != 160
        or 160 * 16 % args.batch_size
        or min(s["pairs"] for s in shards) < 16
    ):
        parser.error("requires all160 train sequences and a batch size dividing2560")
    if args.device == "cuda" and not args.preflight_only and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    project = Path(__file__).resolve().parents[1]
    sources = ["scripts/train_antiuav300_registration_v7.py", "src/aero_ir/utils/manifest.py"] + [
        f"src/aero_ir/registration/{name}.py"
        for name in (
            "protocol_v7",
            "shared_velocity",
            "geometry",
            "mind",
            "protocol_v2",
            "protocol_v4",
            "protocol_v5",
            "qualification_v4",
            "superfusion",
        )
    ]
    spec = {
        "architecture": ARCHITECTURE,
        "arm": args.arm,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "smoke_steps": args.smoke_steps,
        "learning_rate": LEARNING_RATE,
        "mind_weight": MIND_WEIGHT if args.arm == "geometry_mind" else 0,
        "loss_weights": LOSS_WEIGHTS,
        "integration_steps": INTEGRATION_STEPS,
        "initial_checkpoint_sha256": INITIAL_SHA256,
        "cache_file_sha256": file_sha256(cache_path),
        "source_sha256": {s: file_sha256(project / s) for s in sources},
        "fit_split": "train",
        "validation_or_test_access": "none",
        "samples_per_sequence_per_epoch": 16,
        "full_cache_bytes_verified": False,
        "consumed_batches_hashed": True,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": "warn_only_grid_sample_backward_may_be_nondeterministic",
        "generator_training_eligible": "hold_not_qualified",
    }
    if args.preflight_only:
        if args.resume and json.loads((args.output_dir / "run_spec.json").read_text()) != spec:
            parser.error("resume sources, environment or protocol differ from original run")
        print(
            json.dumps(
                {
                    "status": "preflight_only_no_training",
                    "run_spec": spec,
                    "steps_per_epoch": 2560 // args.batch_size,
                },
                indent=2,
            )
        )
        return
    if args.resume:
        if json.loads((args.output_dir / "run_spec.json").read_text()) != spec:
            parser.error("resume sources, environment or protocol differ from original run")
        model, state = load_trained(args.output_dir / "latest.pth", device)
        if state["spec_sha256"] != canonical_hash(spec):
            parser.error("checkpoint protocol hash mismatch")
        start_epoch = state["completed_epochs"]
        if not isinstance(start_epoch, int) or not 0 <= start_epoch <= args.epochs:
            parser.error("invalid completed epoch in resume checkpoint")
        torch.set_rng_state(state["torch_rng"])
        if args.device == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        model = SharedVelocityMatcher(
            load_superfusion_matcher(args.initial_checkpoint, device)
        ).eval()
        start_epoch = 0
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "run_spec.json", spec)
        for name in sources:
            dest = args.output_dir / "sources" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, dest)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    steps_per_epoch = 2560 // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * steps_per_epoch, eta_min=1e-6
    )
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    if args.smoke_steps:
        write_json(
            args.output_dir / "smoke_initial_train_screen.json",
            screen(model, args.cache_root, shards[:1], device),
        )
    if not args.smoke_steps and not (args.output_dir / "initial_train_screen.json").exists():
        if args.resume and start_epoch != 0:
            parser.error("initial screen missing; refusing to relabel trained metrics as initial")
        write_json(
            args.output_dir / "initial_train_screen.json",
            screen(model, args.cache_root, shards, device),
        )
    if not args.resume and not args.smoke_steps:
        save_checkpoint(args.output_dir / "latest.pth", model, optimizer, scheduler, 0, spec)
    attempt = len(list(args.output_dir.glob("attempt_*.jsonl")))
    write_json(
        args.output_dir / f"attempt_{attempt:03d}_runtime.json",
        {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "python": platform.python_version(),
            "resume_from_epoch": start_epoch,
            "spec_sha256": canonical_hash(spec),
        },
    )
    started = time.monotonic()
    updates = 0
    with (args.output_dir / f"attempt_{attempt:03d}.jsonl").open("x") as log:
        for epoch in range(start_epoch, args.epochs):
            selection = epoch_selection(shards, epoch, args.seed)
            totals = {}
            for batch_index, offset in enumerate(range(0, len(selection), args.batch_size)):
                chosen = selection[offset : offset + args.batch_size]
                arrays, digest = read_batch(args.cache_root, chosen)
                vis, ir, vb, ib = tensors(arrays, device)
                observations = None
                if args.arm == "geometry_mind":
                    excluded = [
                        torch.from_numpy(np.stack([automatic_exclusion(x) for x in arrays[k]])).to(
                            device
                        )
                        for k in ARRAY_KEYS[:2]
                    ]
                    observations = (
                        observation(vis, vb, excluded[0]),
                        observation(ir, ib, excluded[1]),
                    )
                values = step(model, optimizer, vis, ir, vb, ib, observations=observations)
                scheduler.step()
                updates += 1
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.0) + value
                record = {
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "loss": values,
                    "selection": chosen,
                    "array_sha256": digest,
                }
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                if batch_index % 10 == 0:
                    print(
                        json.dumps({k: record[k] for k in ("epoch", "batch", "loss")}), flush=True
                    )
                if args.smoke_steps and updates >= args.smoke_steps:
                    write_json(
                        args.output_dir / "smoke_result.json",
                        {
                            "optimizer_steps": updates,
                            "last_loss": values,
                            "checkpoint_is_trained_pilot": False,
                            "device": str(device),
                            "seconds": time.monotonic() - started,
                        },
                    )
                    print("CPU/GPU training smoke complete; NOT a completed pilot", flush=True)
                    return
            save_checkpoint(
                args.output_dir / "latest.pth", model, optimizer, scheduler, epoch + 1, spec
            )
            print(
                json.dumps(
                    {
                        "completed_epoch": epoch + 1,
                        "mean_loss": {k: v / steps_per_epoch for k, v in totals.items()},
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if args.smoke_steps:
        raise ValueError("smoke budget exceeds available training steps")
    if (
        any(file_sha256(project / s) != h for s, h in spec["source_sha256"].items())
        or file_sha256(cache_path) != spec["cache_file_sha256"]
        or file_sha256(args.initial_checkpoint) != INITIAL_SHA256
    ):
        raise ValueError("source/cache manifest changed during run")
    final = args.output_dir / f"shared_velocity_e{args.epochs}.pth"
    if not final.exists():
        shutil.copyfile(args.output_dir / "latest.pth", final)
    elif file_sha256(final) != file_sha256(args.output_dir / "latest.pth"):
        raise ValueError("final checkpoint differs from latest")
    trained, final_state = load_trained(final, device)
    if final_state["completed_epochs"] != args.epochs or final_state[
        "spec_sha256"
    ] != canonical_hash(spec):
        raise ValueError("final checkpoint is incomplete or belongs to another protocol")
    if not (args.output_dir / "final_train_screen.json").exists():
        write_json(
            args.output_dir / "final_train_screen.json",
            {
                **screen(trained, args.cache_root, shards, device),
                "checkpoint_sha256": file_sha256(final),
                "spec_sha256": canonical_hash(spec),
            },
        )
    final_screen = json.loads((args.output_dir / "final_train_screen.json").read_text())
    if (
        final_screen.get("checkpoint_sha256") != file_sha256(final)
        or final_screen.get("spec_sha256") != canonical_hash(spec)
        or final_screen["summary"]["evaluated_pairs"] != 160
    ):
        raise ValueError("final screen does not match the complete trained checkpoint")
    result = {
        "checkpoint_sha256": file_sha256(final),
        "completed_epochs": args.epochs,
        "optimizer_steps": args.epochs * steps_per_epoch,
        "arm": args.arm,
        "generator_training_eligible": "hold_not_qualified",
    }
    if not (args.output_dir / "training_result.json").exists():
        write_json(args.output_dir / "training_result.json", result)
    elif json.loads((args.output_dir / "training_result.json").read_text()) != result:
        raise ValueError("existing training result differs from finalized run")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
