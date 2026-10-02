"""Matched-data continuation versus residual-head pilot. Never generator GO.

Both start from completed v7 geometry and use10 epochs/3200 updates. Optimizer
learning rates differ by parameterization and are explicit, not an isolated
architecture-only ablation. The planned final300-epoch experiment is separate.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from aero_ir.registration.protocol_v7 import SharedVelocityMatcher, step
from aero_ir.registration.residual_velocity import ARCHITECTURE, ResidualVelocityMatcher
from aero_ir.registration.superfusion import DenseMatcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.train_antiuav300_registration_v7 import (
    epoch_selection,
    load_trained,
    read_batch,
    screen,
    tensors,
    write_json,
)
from scripts.verify_registration_v7 import verify_arm, verify_screen, verify_training_trace

ARMS = ("predictor_continuation", "residual_head")
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATES = {"predictor_continuation": 1e-5, "residual_head": 1e-4}
KIND = "matched_v7_continuation_residual_pilot_v1"


def build_model(arm):
    if arm not in ARMS:
        raise ValueError("unknown pilot arm")
    base = SharedVelocityMatcher(DenseMatcher())
    return ResidualVelocityMatcher(base) if arm == "residual_head" else base


def load_checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") != KIND or state.get("arm") not in ARMS:
        raise ValueError("not a residual-comparison pilot checkpoint")
    if any(not torch.isfinite(v).all() for v in state["model_state"].values()):
        raise ValueError("nonfinite model checkpoint")
    model = build_model(state["arm"]).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    return model.eval(), state


def save_checkpoint(path, model, optimizer, scheduler, epoch, spec, device):
    temp = path.with_suffix(".partial")
    torch.save({"kind": KIND, "arm": spec["arm"], "spec_sha256": canonical_hash(spec),
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "completed_epochs": epoch, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "generator_training_eligible": "hold_not_qualified"}, temp)
    os.replace(temp, path)


def verify_products(root, cache, spec):
    """Verify a completed arm, not a training replay or physical qualification."""
    project = Path(__file__).resolve().parents[1]
    for name, digest in spec["source_sha256"].items():
        if file_sha256(project / name) != digest or file_sha256(root / "sources" / name) != digest:
            raise ValueError(f"source drift: {name}")
    if file_sha256(cache / "manifest.json") != spec["cache_file_sha256"]:
        raise ValueError("cache changed")
    shards = json.loads((cache / "manifest.json").read_text())["shards"]
    trace = verify_training_trace(root, spec, shards)
    _, state = load_checkpoint(root / "final.pth", torch.device("cpu"))
    if (state["arm"] != spec["arm"] or state["completed_epochs"] != EPOCHS
            or state["spec_sha256"] != canonical_hash(spec)
            or state["scheduler"]["last_epoch"] != EPOCHS * 320
            or max(int(v["step"]) for v in state["optimizer"]["state"].values()) != EPOCHS * 320):
        raise ValueError("incomplete or mismatched checkpoint/update budget")
    final = json.loads((root / "final_train_screen.json").read_text())
    initial = json.loads((root / "initial_train_screen.json").read_text())
    if (final["checkpoint_sha256"] != file_sha256(root / "final.pth")
            or final["spec_sha256"] != canonical_hash(spec)
            or initial["samples"] != final["samples"]):
        raise ValueError("screen/checkpoint/sample binding differs")
    return {"status": "verified_saved_artifacts_not_replayed", "trace": trace,
            "initial": verify_screen(initial, shards), "final": verify_screen(final, shards),
            "screen_sample_signature": canonical_hash(final["samples"]),
            "final_checkpoint_sha256": file_sha256(root / "final.pth"),
            "generator_training_eligible": "hold_not_qualified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "train", "resume", "verify"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--base-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v7_pilot_e10_seed0"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("experiments/registration_residual_pilot_e10_seed0"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    for value in (args.base_root, args.cache_root, args.output_root):
        if any(c in str(value) for c in ("\n", "\r")):
            parser.error("path contains a newline")
    root = args.output_root / args.arm
    base_root = args.base_root / "geometry"
    proof = verify_arm(base_root, "geometry", args.cache_root)
    if proof["status"] != "verified_saved_artifacts_not_replayed":
        parser.error("verified completed v7 geometry base required")
    project = Path(__file__).resolve().parents[1]
    sources = {**proof["spec"]["source_sha256"]}
    for name in ("scripts/train_registration_residual_pilot.py",
                 "scripts/verify_registration_v7.py",
                 "src/aero_ir/registration/residual_velocity.py"):
        sources[name] = file_sha256(project / name)
    spec = {"kind": KIND, "arm": args.arm, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "seed": 0, "learning_rate": LEARNING_RATES[args.arm], "weight_decay": 1e-5,
            "loss": "unchanged_v7_geometry", "base_normalization_mode": "eval",
            "architecture": (ARCHITECTURE if args.arm == "residual_head"
                             else proof["spec"]["architecture"]),
            "initial_checkpoint_sha256": proof["artifacts_sha256"]["shared_velocity_e10.pth"],
            "source_sha256": sources, "cache_file_sha256": proof["spec"]["cache_file_sha256"],
            "fit_split": "train", "validation_or_test_access": "none",
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "smoke_steps": 2 if args.mode == "smoke" else 0,
            "generator_training_eligible": "hold_not_qualified"}
    if args.mode == "preflight":
        print(json.dumps({"status": "preflight_only_no_training", "spec": spec}, indent=2))
        return
    if args.mode in ("resume", "verify"):
        if json.loads((root / "run_spec.json").read_text()) != spec:
            parser.error("sources, runtime or specification changed; do not overwrite/resume")
        if args.mode == "verify":
            result = verify_products(root, args.cache_root, spec)
            print(json.dumps({k: v for k, v in result.items() if k not in ("initial", "final")},
                             indent=2))
            return
        if (root / "training_result.json").exists():
            result = verify_products(root, args.cache_root, spec)
            if json.loads((root / "training_result.json").read_text()) != result:
                raise ValueError("existing completion record differs")
            print("Already complete and verified; no optimizer update or output overwrite")
            return
    elif root.exists():
        parser.error("output exists; use resume or a fresh output root")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no training directory created")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    start_epoch = 0
    if args.mode == "resume":
        model, state = load_checkpoint(root / "latest.pth", device)
        start_epoch = state["completed_epochs"]
        if (state["spec_sha256"] != canonical_hash(spec)
                or not isinstance(start_epoch, int) or not 0 <= start_epoch <= EPOCHS):
            parser.error("invalid resume checkpoint")
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        base, _ = load_trained(base_root / "shared_velocity_e10.pth", device)
        model = ResidualVelocityMatcher(base).to(device) if args.arm == "residual_head" else base
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "run_spec.json", spec)
        for name, digest in sources.items():
            target = root / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)
            if file_sha256(target) != digest:
                raise ValueError("source changed while snapshotting")
    model.train() if args.arm == "residual_head" else model.eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS * 320, eta_min=spec["learning_rate"] * .1)
    if args.mode == "resume":
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    shards = json.loads((args.cache_root / "manifest.json").read_text())["shards"]
    if not (root / "initial_train_screen.json").exists() and args.mode != "smoke":
        if start_epoch != 0:
            raise ValueError("missing untrained initial screen")
        write_json(root / "initial_train_screen.json",
                   screen(model, args.cache_root, shards, device))
    if args.mode != "resume":
        save_checkpoint(root / "latest.pth", model, optimizer, scheduler, 0, spec, device)
    attempt = len(list(root.glob("attempt_*.jsonl")))
    write_json(root / f"attempt_{attempt:03d}_runtime.json",
               {"spec_sha256": canonical_hash(spec), "resume_from_epoch": start_epoch,
                "device": str(device), "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda" else "CPU"})
    with (root / f"attempt_{attempt:03d}.jsonl").open("x") as log:
        for epoch in range(start_epoch, EPOCHS):
            selection = epoch_selection(shards, epoch, 0)
            for batch, offset in enumerate(range(0, len(selection), BATCH_SIZE), 1):
                chosen = selection[offset:offset+BATCH_SIZE]
                arrays, digest = read_batch(args.cache_root, chosen)
                metrics = step(model, optimizer, *tensors(arrays, device))
                scheduler.step()
                row = {"epoch": epoch+1, "batch": batch, "selection": chosen,
                       "array_sha256": digest, "loss": metrics}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if batch == 1 or batch % 25 == 0:
                    print(json.dumps({"epoch": epoch+1, "batch": batch, "loss": metrics}),
                          flush=True)
                if args.mode == "smoke" and batch == 2:
                    write_json(root / "smoke_result.json",
                               {"optimizer_steps": 2, "last_loss": metrics,
                                "checkpoint_is_completed_pilot": False})
                    print("Two-update training smoke complete; not a completed pilot", flush=True)
                    return
            save_checkpoint(root / "latest.pth", model, optimizer, scheduler, epoch+1, spec, device)
    if file_sha256(base_root / "shared_velocity_e10.pth") != spec["initial_checkpoint_sha256"]:
        raise ValueError("base checkpoint changed")
    final = root / "final.pth"
    if not final.exists():
        shutil.copyfile(root / "latest.pth", final)
    elif file_sha256(final) != file_sha256(root / "latest.pth"):
        raise ValueError("conflicting final checkpoint")
    model, _ = load_checkpoint(final, device)
    if not (root / "final_train_screen.json").exists():
        write_json(root / "final_train_screen.json",
                   {**screen(model, args.cache_root, shards, device),
                    "checkpoint_sha256": file_sha256(final), "spec_sha256": canonical_hash(spec)})
    result = verify_products(root, args.cache_root, spec)
    if not (root / "training_result.json").exists():
        write_json(root / "training_result.json", result)
    elif json.loads((root / "training_result.json").read_text()) != result:
        raise ValueError("existing completion result differs")
    print(json.dumps({"arm": args.arm, "status": result["status"],
                      "final_joint_pass": result["final"]["joint_frame_pass_rate"]}, indent=2))


if __name__ == "__main__":
    main()
