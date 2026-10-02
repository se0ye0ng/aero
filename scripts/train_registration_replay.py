"""Uniform versus failure-aware residual fine-tuning. Never qualification GO.

Both arms start from the same completed residual10 checkpoint and run300 MORE
epochs. This is not a relabelling of its10 epochs, nor a full pipeline result.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch

from aero_ir.registration.protocol_v7 import step
from aero_ir.registration.qualification_v4 import direction_pass, direction_statistics, split_report
from aero_ir.registration.replay_sampling import (
    ARMS,
    BATCH_SIZE,
    UPDATES_PER_EPOCH,
    epoch_selection,
    validate_partition,
)
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.compare_registration_residual_pilot import compare as verify_base
from scripts.train_antiuav300_registration_v7 import read_batch, tensors, write_json
from scripts.train_registration_residual_pilot import build_model
from scripts.train_registration_residual_pilot import load_checkpoint as load_initial
from scripts.verify_registration_v7 import check, is_digest, read_json

KIND = "uniform_failure_aware_residual_finetune_300_v1"
EPOCHS = 300
HOLD = "hold_not_qualified"
EMPTY_CHAIN = "0" * 64
PROJECT = Path(__file__).resolve().parents[1]


def frozen_base_digest(state):
    digest = hashlib.sha256()
    names = sorted(k for k in state if k.startswith("base_model."))
    check(bool(names), "missing frozen base")
    for name in names:
        value = state[name].detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    check(not path.exists(), "refusing to replace a published JSON artifact")
    temporary = path.with_suffix(".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def make_spec(base, cache, arm, smoke=False):
    proof = verify_base(base, cache)
    initial = base / "residual_head" / "final.pth"
    initial_state = torch.load(initial, map_location="cpu", weights_only=True)["model_state"]
    screen = read_json(base / "residual_head" / "final_train_screen.json")
    manifest = read_json(cache / "manifest.json")
    check(
        manifest.get("fit_split") == "train"
        and manifest.get("validation_or_test_access") == "none",
        "not a train-only cache",
    )
    shards = manifest["shards"]
    check(len(shards) == 160, "expected the complete160-sequence train cache")
    partition = {"passing": [], "failing": []}
    for row in screen["rows"]:
        passed = all(direction_pass(row[k]) for k in ("ir_to_rgb_points", "rgb_to_ir_points"))
        partition["passing" if passed else "failing"].append(row["sequence_id"])
    partition = {k: sorted(v) for k, v in partition.items()}
    validate_partition(shards, partition)
    sources = read_json(base / "residual_head" / "run_spec.json")["source_sha256"].copy()
    for name in (
        "scripts/train_registration_replay.py",
        "scripts/run_registration_replay.sh",
        "scripts/compare_registration_residual_pilot.py",
        "src/aero_ir/registration/replay_sampling.py",
    ):
        sources[name] = file_sha256(PROJECT / name)
    spec = {
        "kind": KIND,
        "arm": arm,
        "epochs": 1 if smoke else EPOCHS,
        "updates_per_epoch": 2 if smoke else UPDATES_PER_EPOCH,
        "smoke": smoke,
        "seed": 0,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "eta_min": 1e-5,
        "loss": "unchanged_v7_geometry",
        "trainable": "residual_head_only",
        "initial_checkpoint_sha256": file_sha256(initial),
        "frozen_base_sha256": frozen_base_digest(initial_state),
        "initial_screen_sha256": file_sha256(base / "residual_head" / "final_train_screen.json"),
        "cache_file_sha256": file_sha256(cache / "manifest.json"),
        "source_sha256": sources,
        "partition": partition,
        "fit_split": "train",
        "validation_or_test_access": "none",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "generator_training_eligible": HOLD,
    }
    check(
        spec["initial_checkpoint_sha256"]
        == proof["arms"]["residual_head"]["final_checkpoint_sha256"],
        "initial checkpoint changed",
    )
    return spec, shards, initial


def schedule(spec, shards, epoch):
    return epoch_selection(shards, spec["partition"], spec["arm"], epoch, spec["seed"])[
        : spec["updates_per_epoch"] * spec["batch_size"]
    ]


def verify_trace(root, spec, shards, expected_updates):
    """Validate epoch-boundary rollback, exact samples and committed row chain.

    Recorded array hashes are not a fresh decode/replay of the entire cache.
    Only the next attempt's explicit checkpoint boundary discards interrupted work.
    """
    chains, abandoned = [], 0
    logs = sorted(root.glob("attempt_*.jsonl"))
    check(bool(logs), "no attempt logs")
    per_epoch = spec["updates_per_epoch"]
    current_epoch, selection = None, None
    for attempt, log in enumerate(logs):
        check(log.name == f"attempt_{attempt:03d}.jsonl", "missing attempt index")
        runtime = read_json(log.with_name(log.stem + "_runtime.json"))
        start = runtime["resume_from_epoch"]
        check(type(start) is int and 0 <= start <= spec["epochs"], "invalid resume epoch")
        check(runtime["spec_sha256"] == canonical_hash(spec), "attempt protocol mismatch")
        prefix = start * per_epoch
        check(prefix <= len(chains), "checkpoint claims unlogged updates")
        abandoned += len(chains) - prefix
        chains = chains[:prefix]
        lines = log.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if attempt < len(logs) - 1 and i == len(lines) - 1:
                    abandoned += 1
                    break
                raise ValueError(
                    "incomplete final journal; resume from latest checkpoint"
                ) from None
            epoch, batch = divmod(prefix + i, per_epoch)
            check(
                epoch < spec["epochs"] and row["epoch"] == epoch + 1 and row["batch"] == batch + 1,
                "out-of-order or extra update",
            )
            if current_epoch != epoch:
                selection = schedule(spec, shards, epoch)
                current_epoch = epoch
            offset = batch * spec["batch_size"]
            check(
                row["selection"]
                == [list(s) for s in selection[offset : offset + spec["batch_size"]]],
                "selection differs from frozen sampler",
            )
            check(is_digest(row["array_sha256"]), "invalid consumed-array hash")
            check(
                {"total", "gradient_norm"} <= row["loss"].keys()
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in row["loss"].values()
                ),
                "invalid training metric",
            )
            previous = chains[-1] if chains else EMPTY_CHAIN
            chain = canonical_hash({"previous": previous, "row": row})
            chains.append(chain)
    check(len(chains) == expected_updates, "incomplete committed budget")
    return {
        "verified_optimizer_steps": len(chains),
        "abandoned_logged_steps": abandoned,
        "attempts": len(logs),
        "committed_chain": chains[-1] if chains else EMPTY_CHAIN,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, spec, device, chain):
    temp = path.with_suffix(".partial")
    torch.save(
        {
            "kind": KIND,
            "spec_sha256": canonical_hash(spec),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_epochs": epoch,
            "committed_chain": chain,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "generator_training_eligible": HOLD,
        },
        temp,
    )
    os.replace(temp, path)


def load_checkpoint(path, spec, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    check(
        state.get("kind") == KIND and state.get("spec_sha256") == canonical_hash(spec),
        "checkpoint protocol mismatch",
    )
    epoch = state["completed_epochs"]
    check(type(epoch) is int and 0 <= epoch <= spec["epochs"], "invalid checkpoint epoch")
    updates = epoch * spec["updates_per_epoch"]
    check(state["scheduler"]["last_epoch"] == updates, "scheduler/update mismatch")
    steps = [int(v["step"]) for v in state["optimizer"]["state"].values()]
    check(
        (not steps and updates == 0) or (bool(steps) and set(steps) == {updates}),
        "optimizer/update mismatch",
    )
    check(is_digest(state["committed_chain"]), "invalid journal binding")
    check(
        all(torch.isfinite(v).all() for v in state["model_state"].values()), "nonfinite checkpoint"
    )
    if "frozen_base_sha256" in spec:
        check(
            frozen_base_digest(state["model_state"]) == spec["frozen_base_sha256"],
            "frozen base changed during training",
        )
    model = build_model("residual_head").to(device)
    model.load_state_dict(state["model_state"], strict=True)
    return model.eval(), state


def evaluate(model, cache, shards, device, divisor):
    rows, samples = [], []
    model.eval()
    with torch.no_grad():
        for shard in sorted(shards, key=lambda s: s["sequence_id"]):
            selected = [(shard["sequence_id"], shard["pairs"] // divisor)]
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
        "frame_divisor": divisor,
        "evaluated_split": "train",
        "not_independent_of_training": True,
        "generator_training_eligible": HOLD,
    }


def check_sources(root, cache, spec):
    check(file_sha256(cache / "manifest.json") == spec["cache_file_sha256"], "cache drift")
    for name, digest in spec["source_sha256"].items():
        check(
            file_sha256(PROJECT / name) == digest
            and file_sha256(root / "sources" / name) == digest,
            f"source drift: {name}",
        )


def verify_products(root, cache, spec, shards):
    check(
        not spec["smoke"]
        and spec["epochs"] == EPOCHS
        and spec["updates_per_epoch"] == UPDATES_PER_EPOCH,
        "not a full300-epoch run",
    )
    check_sources(root, cache, spec)
    model, state = load_checkpoint(root / "final.pth", spec, torch.device("cpu"))
    del model
    check(state["completed_epochs"] == EPOCHS, "training incomplete")
    trace = verify_trace(root, spec, shards, EPOCHS * UPDATES_PER_EPOCH)
    check(trace["committed_chain"] == state["committed_chain"], "checkpoint/journal mismatch")
    digest = file_sha256(root / "final.pth")
    result = {
        "status": "saved_artifacts_verified_not_replayed",
        "trace": trace,
        "checkpoint_sha256": digest,
        "panels": {},
        "generator_training_eligible": HOLD,
    }
    for divisor in (2, 4):
        record = read_json(root / f"final_train_div{divisor}.json")
        check(
            record["checkpoint_sha256"] == digest and record["spec_sha256"] == canonical_hash(spec),
            "screen/checkpoint mismatch",
        )
        check(
            record["evaluated_split"] == "train"
            and record["generator_training_eligible"] == HOLD
            and record["frame_divisor"] == divisor,
            "invalid screen scope",
        )
        ordered = sorted(shards, key=lambda s: s["sequence_id"])
        check(
            [r["sequence_id"] for r in record["rows"]] == [s["sequence_id"] for s in ordered],
            "incomplete screen rows",
        )
        check(len(record["samples"]) == len(ordered), "incomplete screen samples")
        for s, sample in zip(ordered, record["samples"], strict=True):
            check(
                sample["selection"] == [[s["sequence_id"], s["pairs"] // divisor]]
                and is_digest(sample["array_sha256"]),
                "screen sample mismatch",
            )
        summary = split_report(record["rows"])
        check(summary == record["summary"], "stored summary disagrees with predicates")
        result["panels"][str(divisor)] = summary
    return result


def compare(output, cache, base):
    specs, products, panels = {}, {}, {}
    for arm in ARMS:
        spec, shards, _ = make_spec(base, cache, arm)
        root = output / arm
        check(read_json(root / "run_spec.json") == spec, "run specification changed")
        specs[arm] = spec
        products[arm] = verify_products(root, cache, spec, shards)
        check(
            read_json(root / "training_result.json") == products[arm], "completion record mismatch"
        )
        panels[arm] = {d: read_json(root / f"final_train_div{d}.json") for d in (2, 4)}
    check(
        {k: v for k, v in specs[ARMS[0]].items() if k != "arm"}
        == {k: v for k, v in specs[ARMS[1]].items() if k != "arm"},
        "unmatched protocols",
    )
    paired = {}
    for d in (2, 4):
        a, b = (panels[arm][d] for arm in ARMS)
        check(a["samples"] == b["samples"], "different evaluation input bytes")

        def passed(r):
            return all(direction_pass(r[k]) for k in ("ir_to_rgb_points", "rgb_to_ir_points"))

        paired[str(d)] = {"failure_aware_only": [], "uniform_only": []}
        for left, right in zip(a["rows"], b["rows"], strict=True):
            if passed(left) != passed(right):
                key = "uniform_only" if passed(left) else "failure_aware_only"
                paired[str(d)][key].append(left["sequence_id"])
    return {
        "kind": KIND,
        "arms": products,
        "paired": paired,
        "matched_initialization_loss_and_budget": True,
        "matched_training_data": False,
        "model_inference_replayed": False,
        "generator_training_eligible": HOLD,
        "limitations": [
            "Train midpoint and quarter screens, not physical accuracy or held-out tests.",
            "300 additional epochs after the original10, not an entire pipeline result.",
        ],
    }


def run(args, spec, shards, initial):
    root = args.output_root / args.arm
    if args.mode in ("resume", "verify"):
        check(
            read_json(root / "run_spec.json") == spec,
            "protocol/runtime/source drift; cannot resume",
        )
        check_sources(root, args.cache_root, spec)
        if args.mode == "verify" or (root / "training_result.json").exists():
            result = verify_products(root, args.cache_root, spec, shards)
            check(read_json(root / "training_result.json") == result, "completion record mismatch")
            print("Already complete and verified; no training or overwrite", flush=True)
            return
    else:
        check(not root.exists(), "output exists; use resume or a fresh output root")
    device = torch.device(args.device)
    check(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable; no run created")
    torch.set_num_threads(1)
    torch.manual_seed(spec["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    start, chain = 0, EMPTY_CHAIN
    if args.mode == "resume":
        model, state = load_checkpoint(root / "latest.pth", spec, device)
        start, chain = state["completed_epochs"], state["committed_chain"]
    else:
        model, _ = load_initial(initial, device)
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "run_spec.json", spec)
        for name in spec["source_sha256"]:
            target = root / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, target)
        check_sources(root, args.cache_root, spec)
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, spec["epochs"] * spec["updates_per_epoch"], eta_min=spec["eta_min"]
    )
    if args.mode == "resume":
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        save_checkpoint(root / "latest.pth", model, optimizer, scheduler, 0, spec, device, chain)
    attempt = len(list(root.glob("attempt_*.jsonl")))
    write_json(
        root / f"attempt_{attempt:03d}_runtime.json",
        {
            "spec_sha256": canonical_hash(spec),
            "resume_from_epoch": start,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "hostname": os.uname().nodename,
        },
    )
    with (root / f"attempt_{attempt:03d}.jsonl").open("x") as log:
        # Empty new attempt explicitly supersedes uncommitted interrupted updates.
        trace = verify_trace(root, spec, shards, start * spec["updates_per_epoch"])
        check(trace["committed_chain"] == chain, "resume checkpoint/journal mismatch")
        model.train()
        for epoch in range(start, spec["epochs"]):
            selection = schedule(spec, shards, epoch)
            for batch, offset in enumerate(range(0, len(selection), BATCH_SIZE), 1):
                chosen = selection[offset : offset + BATCH_SIZE]
                arrays, digest = read_batch(args.cache_root, chosen)
                metrics = step(model, optimizer, *tensors(arrays, device))
                scheduler.step()
                row = {
                    "epoch": epoch + 1,
                    "batch": batch,
                    "selection": chosen,
                    "array_sha256": digest,
                    "loss": metrics,
                }
                chain = canonical_hash({"previous": chain, "row": row})
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if batch == 1 or batch % 25 == 0:
                    print(
                        json.dumps(
                            {"arm": args.arm, "epoch": epoch + 1, "batch": batch, "loss": metrics}
                        ),
                        flush=True,
                    )
            os.fsync(log.fileno())
            save_checkpoint(
                root / "latest.pth", model, optimizer, scheduler, epoch + 1, spec, device, chain
            )
    check_sources(root, args.cache_root, spec)
    check(file_sha256(initial) == spec["initial_checkpoint_sha256"], "initializer drift")
    if spec["smoke"]:
        restored, saved = load_checkpoint(root / "latest.pth", spec, device)
        check(
            all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()),
            "smoke checkpoint reload differs",
        )
        trace = verify_trace(root, spec, shards, 2)
        check(trace["committed_chain"] == saved["committed_chain"], "smoke trace mismatch")
        write_json(
            root / "smoke_result.json",
            {"trace": trace, "checkpoint_reload_equal": True, "generator_training_eligible": HOLD},
        )
        print("Two-update smoke complete; NOT the300-epoch experiment", flush=True)
        return
    final = root / "final.pth"
    if not final.exists():
        temporary = final.with_suffix(".partial")
        shutil.copyfile(root / "latest.pth", temporary)
        os.replace(temporary, final)
    check(file_sha256(final) == file_sha256(root / "latest.pth"), "conflicting final checkpoint")
    model, _ = load_checkpoint(final, spec, device)
    for divisor in (2, 4):
        target = root / f"final_train_div{divisor}.json"
        if not target.exists():
            record = {
                **evaluate(model, args.cache_root, shards, device, divisor),
                "checkpoint_sha256": file_sha256(final),
                "spec_sha256": canonical_hash(spec),
            }
            atomic_json(target, record)
    result = verify_products(root, args.cache_root, spec, shards)
    atomic_json(root / "training_result.json", result)
    print(
        json.dumps(
            {
                "status": "300_epoch_finetune_complete_not_qualified",
                "panels": {d: v["joint_frame_pass_rate"] for d, v in result["panels"].items()},
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "smoke", "train", "resume", "verify", "compare")
    )
    parser.add_argument("--arm", choices=ARMS, default="uniform")
    parser.add_argument(
        "--base-root", type=Path, default=Path("experiments/registration_residual_pilot_e10_seed0")
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("experiments/registration_replay_e300_seed0")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    for value in (args.base_root, args.cache_root, args.output_root):
        check(not any(c in str(value) for c in ("\n", "\r")), "path contains a newline")
    if args.mode == "compare":
        result = compare(args.output_root, args.cache_root, args.base_root)
        path = args.output_root / "comparison.json"
        if path.exists():
            check(read_json(path) == result, "existing comparison differs")
        else:
            atomic_json(path, result)
        print(f"Verified comparison: {path}; qualification remains HOLD")
        return
    spec, shards, initial = make_spec(
        args.base_root, args.cache_root, args.arm, args.mode == "smoke"
    )
    if args.mode == "preflight":
        print(json.dumps({"status": "preflight_only", "spec": spec}, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Advisory kernel lock prevents two jobs/resumes writing the same arm.
    with (args.output_root / f".{args.arm}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another process holds this arm's training lock") from None
        run(args, spec, shards, initial)


if __name__ == "__main__":
    main()
