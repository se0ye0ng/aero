"""Read-only audit of the two completed v7 pilots; never grants generator GO.

Checks saved evidence and recomputes gates from rows, not model inference. It is
neither a training replay nor independent correspondence accuracy measurement.
"""

import argparse
import json
import math
from pathlib import Path

from aero_ir.registration.protocol_v7 import ARCHITECTURE, LOSS_WEIGHTS, MIND_WEIGHT
from aero_ir.registration.qualification_v4 import THRESHOLDS, direction_pass, split_report
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.train_antiuav300_registration_v7 import (
    INITIAL_SHA256,
    epoch_selection,
    load_trained,
)

ARMS = ("geometry", "geometry_mind")


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON value in {path}: {value}")

    return json.loads(path.read_text(), parse_constant=reject)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_screen(screen, shards):
    check(
        screen.get("evaluated_split") == "train"
        and screen.get("validation_or_test_access") == "none",
        "screen is not train-only",
    )
    check(
        screen.get("generator_training_eligible") == "hold_not_qualified",
        "train screen must not grant generator qualification",
    )
    ids = [r["sequence_id"] for r in screen["rows"]]
    expected = sorted(s["sequence_id"] for s in shards)
    check(
        len(ids) == len(set(ids)) == len(expected) and sorted(ids) == expected,
        "missing, duplicated or unexpected screen sequences",
    )
    samples = screen["samples"]
    positions = {s["sequence_id"]: s["pairs"] // 2 for s in shards}
    check(len(samples) == len(ids), "screen sample count differs from rows")
    for sid, sample in zip(ids, samples, strict=True):
        check(sample["selection"] == [[sid, positions[sid]]], "screen frame identity differs")
        check(is_digest(sample["array_sha256"]), "invalid screen array hash")
    recomputed = split_report(screen["rows"])
    check(recomputed == screen["summary"], "stored screen summary differs from recomputed gates")
    return recomputed


def is_digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def verify_training_trace(root, spec, shards):
    """Reconstruct epoch-boundary resume lineage; do not count abandoned updates."""
    logs = sorted(root.glob("attempt_*.jsonl"))
    check(bool(logs), "no training attempt logs")
    per_epoch = len(shards) * 16 // spec["batch_size"]
    retained, discarded = [], 0
    spec_hash = canonical_hash(spec)
    schedules = {}
    for attempt, log in enumerate(logs):
        check(log.name == f"attempt_{attempt:03d}.jsonl", "missing attempt index")
        runtime = read_json(log.with_name(log.stem + "_runtime.json"))
        check(runtime["spec_sha256"] == spec_hash, "attempt protocol hash mismatch")
        start = runtime["resume_from_epoch"]
        check(
            isinstance(start, int) and not isinstance(start, bool) and 0 <= start <= spec["epochs"],
            "invalid resume epoch",
        )
        prefix = start * per_epoch
        check(prefix <= len(retained), "resume checkpoint claims epochs absent from previous logs")
        discarded += len(retained) - prefix
        retained = retained[:prefix]
        lines = log.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Only a superseded interrupted attempt may have one incomplete last line.
                if attempt < len(logs) - 1 and i == len(lines) - 1:
                    discarded += 1
                    break
                raise ValueError("incomplete final training log") from None
            index = prefix + i
            epoch, batch = divmod(index, per_epoch)
            check(
                epoch < spec["epochs"] and row["epoch"] == epoch + 1 and row["batch"] == batch + 1,
                "missing, duplicated or out-of-order optimizer step",
            )
            if epoch not in schedules:
                schedules[epoch] = epoch_selection(shards, epoch, spec["seed"])
            offset = batch * spec["batch_size"]
            expected = [list(v) for v in schedules[epoch][offset : offset + spec["batch_size"]]]
            check(
                row["selection"] == expected,
                "training sample selection differs from frozen sampler",
            )
            check(is_digest(row["array_sha256"]), "invalid consumed-batch hash")
            check(
                bool(row["loss"])
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in row["loss"].values()
                ),
                "nonfinite training metric",
            )
            check(
                "total" in row["loss"] and "gradient_norm" in row["loss"],
                "missing training metrics",
            )
            if spec["arm"] == "geometry_mind":
                check(
                    all(
                        k in row["loss"]
                        for k in ("mind", "mind_eligible_fraction", "mind_source_observed_fraction")
                    ),
                    "MIND arm did not report visual observations",
                )
            else:
                check("mind" not in row["loss"], "geometry control unexpectedly contains MIND")
            retained.append(row)
    check(
        len(retained) == spec["epochs"] * per_epoch, "training trace is not a full requested budget"
    )
    signature = canonical_hash(
        [{k: r[k] for k in ("epoch", "batch", "selection", "array_sha256")} for r in retained]
    )
    summary = {
        "verified_optimizer_steps": len(retained),
        "abandoned_logged_steps": discarded,
        "consumed_data_signature": signature,
        "attempts": len(logs),
    }
    if spec["arm"] == "geometry_mind":
        summary["mean_mind_eligible_fraction"] = sum(
            r["loss"]["mind_eligible_fraction"] for r in retained
        ) / len(retained)
        summary["mean_mind_source_observed_fraction"] = sum(
            r["loss"]["mind_source_observed_fraction"] for r in retained
        ) / len(retained)
    return summary


def verify_arm(root, arm, cache):
    import torch

    required = [
        "run_spec.json",
        "training_result.json",
        "initial_train_screen.json",
        "final_train_screen.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {
            "status": "pending_or_incomplete",
            "missing": missing,
            "process_state": "not_inferred",
        }
    spec = read_json(root / "run_spec.json")
    expected = {
        "architecture": ARCHITECTURE,
        "arm": arm,
        "epochs": 10,
        "batch_size": 8,
        "seed": 0,
        "smoke_steps": 0,
        "samples_per_sequence_per_epoch": 16,
        "fit_split": "train",
        "validation_or_test_access": "none",
        "initial_checkpoint_sha256": INITIAL_SHA256,
        "integration_steps": 7,
        "loss_weights": LOSS_WEIGHTS,
        "learning_rate": 1e-5,
        "mind_weight": MIND_WEIGHT if arm == "geometry_mind" else 0,
    }
    for key, value in expected.items():
        check(spec.get(key) == value, f"unexpected pilot specification: {key}")
    check(
        spec.get("generator_training_eligible") == "hold_not_qualified",
        "pilot must remain unqualified",
    )
    required_sources = {
        "scripts/train_antiuav300_registration_v7.py",
        "src/aero_ir/registration/protocol_v7.py",
        "src/aero_ir/registration/shared_velocity.py",
        "src/aero_ir/registration/qualification_v4.py",
        "src/aero_ir/registration/geometry.py",
        "src/aero_ir/registration/protocol_v2.py",
    }
    check(required_sources <= spec["source_sha256"].keys(), "incomplete source provenance")
    project = Path(__file__).resolve().parents[1]
    for name, digest in spec["source_sha256"].items():
        snapshot = (root / "sources" / name).resolve()
        check(snapshot.is_relative_to((root / "sources").resolve()), "source path outside archive")
        check(file_sha256(snapshot) == digest, f"changed archived source: {name}")
        check(
            file_sha256(project / name) == digest,
            f"current verifier dependency differs from run: {name}",
        )
    manifest_path = cache / "manifest.json"
    check(file_sha256(manifest_path) == spec["cache_file_sha256"], "cache manifest changed")
    manifest = read_json(manifest_path)
    check(
        canonical_hash({k: v for k, v in manifest.items() if k != "cache_manifest_sha256"})
        == manifest["cache_manifest_sha256"],
        "invalid cache content hash",
    )
    check(
        manifest["fit_split"] == "train" and manifest["validation_or_test_access"] == "none",
        "not a train cache",
    )
    shards = manifest["shards"]
    check(
        len(shards) == len({s["sequence_id"] for s in shards}) == 160,
        "incomplete official train panel",
    )
    trace = verify_training_trace(root, spec, shards)
    checkpoint = root / "shared_velocity_e10.pth"
    digest = file_sha256(checkpoint)
    model, state = load_trained(checkpoint, torch.device("cpu"))
    del model
    check(
        state["spec_sha256"] == canonical_hash(spec) and state["completed_epochs"] == 10,
        "checkpoint protocol or epoch mismatch",
    )
    check(
        state["scheduler"]["last_epoch"] == trace["verified_optimizer_steps"],
        "scheduler budget mismatch",
    )
    steps = [int(v["step"]) for v in state["optimizer"]["state"].values() if "step" in v]
    check(
        bool(steps) and max(steps) == trace["verified_optimizer_steps"], "optimizer budget mismatch"
    )
    result = read_json(root / "training_result.json")
    check(
        result
        == {
            "checkpoint_sha256": digest,
            "completed_epochs": 10,
            "optimizer_steps": trace["verified_optimizer_steps"],
            "arm": arm,
            "generator_training_eligible": "hold_not_qualified",
        },
        "completion record mismatch",
    )
    initial = read_json(root / "initial_train_screen.json")
    final = read_json(root / "final_train_screen.json")
    before, after = verify_screen(initial, shards), verify_screen(final, shards)
    check(initial["samples"] == final["samples"], "before/after sample content differs")
    check(
        final["checkpoint_sha256"] == digest and final["spec_sha256"] == canonical_hash(spec),
        "final screen checkpoint binding mismatch",
    )
    improved, regressed = [], []
    first = {r["sequence_id"]: r for r in initial["rows"]}
    for row in final["rows"]:
        old = first[row["sequence_id"]]
        old_pass = all(direction_pass(old[k]) for k in ("ir_to_rgb_points", "rgb_to_ir_points"))
        new_pass = all(direction_pass(row[k]) for k in ("ir_to_rgb_points", "rgb_to_ir_points"))
        if new_pass and not old_pass:
            improved.append(row["sequence_id"])
        if old_pass and not new_pass:
            regressed.append(row["sequence_id"])
    return {
        "status": "verified_saved_artifacts_not_replayed",
        "spec": spec,
        "trace": trace,
        "initial": before,
        "final": after,
        "new_pass_sequences": improved,
        "new_failure_sequences": regressed,
        "screen_sample_signature": canonical_hash(final["samples"]),
        "artifacts_sha256": {
            name: file_sha256(root / name) for name in [*required, checkpoint.name]
        },
        "train_geometry_threshold_met": after["joint_frame_pass_rate"]
        >= THRESHOLDS["minimum_joint_frame_pass_rate"]
        and after["sequence_macro_pass_rate"] >= THRESHOLDS["minimum_sequence_macro_pass_rate"],
    }


def compare_arms(arms):
    check(
        all(arms[a]["status"] == "verified_saved_artifacts_not_replayed" for a in ARMS),
        "both arms must be complete and verified",
    )
    left, right = (arms[a] for a in ARMS)
    ignored = {"arm", "mind_weight"}
    check(
        {k: v for k, v in left["spec"].items() if k not in ignored}
        == {k: v for k, v in right["spec"].items() if k not in ignored},
        "unmatched protocol or source versions",
    )
    check(
        left["trace"]["consumed_data_signature"] == right["trace"]["consumed_data_signature"],
        "arms consumed different training frames, order or bytes",
    )
    check(
        left["screen_sample_signature"] == right["screen_sample_signature"],
        "different screen observations",
    )
    return {
        "matched_budget_and_data": True,
        "mind_minus_geometry_joint_pass_rate": right["final"]["joint_frame_pass_rate"]
        - left["final"]["joint_frame_pass_rate"],
        "candidates_for_heldout_engineering_screen": [
            a for a in ARMS if arms[a]["train_geometry_threshold_met"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v7_pilot_e10_seed0"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite audit report")
    arms = {}
    for arm in ARMS:
        try:
            arms[arm] = verify_arm(args.run_root / arm, arm, args.cache_root)
        except (ValueError, KeyError, OSError, RuntimeError, TypeError) as error:
            arms[arm] = {"status": "verification_failed", "error": str(error)}
    comparison = None
    comparison_error = None
    if all(v["status"] == "verified_saved_artifacts_not_replayed" for v in arms.values()):
        try:
            comparison = compare_arms(arms)
        except ValueError as error:
            comparison_error = str(error)
    report = {
        "kind": "v7_saved_pilot_evidence_audit",
        "arms": arms,
        "comparison": comparison,
        "comparison_error": comparison_error,
        "model_inference_replayed": False,
        "cache_array_bytes_rehashed": False,
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "Train screens are not held-out physical accuracy.",
            "Batch hashes are compared across arms, not rederived from all cache arrays.",
            "Initial screen has no independent checkpoint replay.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "arms": {k: v["status"] for k, v in arms.items()},
                "comparison": comparison,
                "report": str(args.out),
            },
            indent=2,
        )
    )
    if comparison is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
