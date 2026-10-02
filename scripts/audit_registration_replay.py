"""Native-input engineering audit of the300-additional-epoch replay comparison.

Require both completed, verified arms and at least one frozen95% train-midpoint
candidate before selecting train/validation observations. Never generator GO.
"""

import argparse
import json
import math
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch

from aero_ir.registration.geometry import CONVENTION
from aero_ir.registration.qualification_v4 import THRESHOLDS, direction_pass, gate_report
from aero_ir.registration.replay_sampling import ARMS
from aero_ir.utils.manifest import canonical_hash, file_sha256
from aero_ir.utils.paths import antiuav300_root
from scripts.audit_antiuav300_registration_v7 import (
    SPLITS,
    Coverage,
    _batches,
    _iter_pairs,
    build_panel,
    check,
    infer_rows,
    source_hashes,
)
from scripts.train_registration_replay import HOLD, atomic_json, compare, load_checkpoint
from scripts.verify_registration_v7 import read_json

PROJECT = Path(__file__).resolve().parents[1]


def candidates(comparison):
    def valid_rate(value):
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1

    return [
        arm
        for arm in ARMS
        if all(
            valid_rate(comparison["arms"][arm]["panels"]["2"][key])
            for key in ("joint_frame_pass_rate", "sequence_macro_pass_rate")
        )
        and comparison["arms"][arm]["panels"]["2"]["joint_frame_pass_rate"]
        >= THRESHOLDS["minimum_joint_frame_pass_rate"]
        and comparison["arms"][arm]["panels"]["2"]["sequence_macro_pass_rate"]
        >= THRESHOLDS["minimum_sequence_macro_pass_rate"]
    ]


def artifact_hashes(root):
    result = {}
    for arm in ARMS:
        folder = root / arm
        paths = [
            folder / name
            for name in (
                "run_spec.json",
                "training_result.json",
                "final.pth",
                "final_train_div2.json",
                "final_train_div4.json",
            )
        ]
        paths += sorted(folder.glob("attempt_*.jsonl"))
        paths += sorted(folder.glob("attempt_*_runtime.json"))
        result[arm] = {str(p.relative_to(folder)): file_sha256(p) for p in paths}
    return result


def audit_sources():
    sources = source_hashes(PROJECT)
    for name in (
        "scripts/audit_registration_replay.py",
        "scripts/run_registration_replay_audit.sh",
        "scripts/train_registration_replay.py",
        "scripts/train_registration_residual_pilot.py",
        "scripts/compare_registration_residual_pilot.py",
    ):
        sources[name] = file_sha256(PROJECT / name)
    return sources


def paired_key(left, right):
    for key in ("split", "sequence_id", "frame_index", "array_sha256"):
        check(left[key] == right[key], "arm observations differ")
    flags = tuple(
        all(direction_pass(row[k]) for k in ("ir_to_rgb_points", "rgb_to_ir_points"))
        for row in (left, right)
    )
    return {
        (True, True): "both_pass",
        (True, False): "uniform_only_pass",
        (False, True): "failure_aware_only_pass",
        (False, False): "neither_pass",
    }[flags]


def run(args):
    check(args.batch_size > 0, "batch size must be positive")
    check(args.samples_per_sequence in (0, 8), "use eight-frame or exhaustive eligible panel")
    for name in ("root", "run_root", "base_root", "cache_root", "out_dir"):
        check(not any(c in str(getattr(args, name)) for c in ("\n", "\r")), f"newline in {name}")
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite audit directory; use a fresh --out-dir")
    sources = audit_sources()
    comparison, identities, eligible, error = None, None, [], None
    try:
        identities = artifact_hashes(args.run_root)
        comparison = compare(args.run_root, args.cache_root, args.base_root)
        check(
            artifact_hashes(args.run_root) == identities, "training artifacts changed in preflight"
        )
        eligible = candidates(comparison)
    except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    ready = bool(eligible) and error is None
    plan = {
        "schema_version": 1,
        "kind": "replay300_engineering_audit_preflight",
        "ready_for_engineering_inference": ready,
        "candidates_from_train_screen": eligible,
        "comparison": comparison,
        "training_artifacts_sha256": identities,
        "blocked_reason": None
        if ready
        else error or "Neither completed arm reached both frozen95% train-midpoint rates.",
        "source_sha256": sources,
        "coordinate_convention": CONVENTION,
        "thresholds": THRESHOLDS,
        "pixel_inference_executed": False,
        "training_process_state": "not_inferred_from_files",
        "generator_training_eligible": HOLD,
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = args.out_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
        check(file_sha256(target) == digest, "source changed while snapshotting")
    if not ready:
        atomic_json(args.out_dir / "preflight.json", plan)
        print(json.dumps({"ready": False, "reason": plan["blocked_reason"]}))
        return 2
    # No validation annotations, videos or CUDA queries are accessed above.
    expected, panel = build_panel(args.root, args.cache_root, args.samples_per_sequence)
    panel["kind"] = "replay300_frozen_annotation_selected_evaluation_panel"
    plan["panel_sha256"] = canonical_hash(panel)
    plan["cache_manifest_sha256"] = file_sha256(args.cache_root / "manifest.json")
    plan["selected_pairs"] = {split: len(values) for split, values in expected.items()}
    atomic_json(args.out_dir / "panel.json", panel)
    atomic_json(args.out_dir / "preflight.json", plan)
    if args.preflight_only:
        print(json.dumps({"ready": True, "selected_pairs": plan["selected_pairs"]}))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; preflight preserved, no inference started")
    models = {}
    for arm in ARMS:
        folder = args.run_root / arm
        checkpoint = folder / "final.pth"
        check(file_sha256(checkpoint) == identities[arm]["final.pth"], "checkpoint changed")
        spec = read_json(folder / "run_spec.json")
        check(spec["arm"] == arm, "checkpoint arm differs")
        models[arm], _ = load_checkpoint(checkpoint, spec, device)
    coverage = {arm: Coverage(expected) for arm in ARMS}
    outcomes = {
        split: dict.fromkeys(
            ("both_pass", "uniform_only_pass", "failure_aware_only_pass", "neither_pass"), 0
        )
        for split in SPLITS
    }
    with ExitStack() as stack:
        outputs = {
            arm: stack.enter_context((args.out_dir / f"{arm}.jsonl").open("x")) for arm in ARMS
        }
        for batch in _batches(
            _iter_pairs(args.root, SPLITS, args.samples_per_sequence), args.batch_size
        ):
            predictions = {arm: infer_rows(models[arm], batch, device) for arm in ARMS}
            for arm, rows in predictions.items():
                for row in rows:
                    coverage[arm].add(row)
                    outputs[arm].write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                outputs[arm].flush()
            for left, right in zip(predictions[ARMS[0]], predictions[ARMS[1]], strict=True):
                outcomes[left["split"]][paired_key(left, right)] += 1
            print(
                f"Audited {sum(len(v) for v in coverage[ARMS[0]].seen.values())} pairs", flush=True
            )
    check(audit_sources() == sources, "audit source drift during inference")
    check(artifact_hashes(args.run_root) == identities, "training artifact drift during inference")
    check(
        file_sha256(args.cache_root / "manifest.json") == plan["cache_manifest_sha256"],
        "cache manifest drift during inference",
    )
    for name, digest in panel["annotation_sha256"].items():
        check(file_sha256(args.root / name) == digest, "annotation drift during inference")
    metrics = {arm: coverage[arm].finish() for arm in ARMS}
    signatures = {arm: coverage[arm].data_signature.hexdigest() for arm in ARMS}
    check(len(set(signatures.values())) == 1, "different consumed evaluation observations")
    complete = not any(panel["sequences_without_eligible_pairs"].values())
    report = {
        "schema_version": 1,
        "kind": "matched_budget_replay300_engineering_inference_audit",
        "panel_sha256": file_sha256(args.out_dir / "panel.json"),
        "preflight_sha256": file_sha256(args.out_dir / "preflight.json"),
        "row_artifacts_sha256": {arm: file_sha256(args.out_dir / f"{arm}.jsonl") for arm in ARMS},
        "source_sha256": sources,
        "coordinate_convention": CONVENTION,
        "field_conversion": "direct centre fields from model.fields; no raw-field adapter",
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "paired_outcomes": outcomes,
        "consumed_pair_signatures": signatures,
        "exact_eligible_pair_coverage": True,
        "all_official_sequences_have_observations": complete,
        "gates": {
            arm: gate_report(
                metrics[arm], exhaustive=args.samples_per_sequence == 0, complete_coverage=complete
            )
            for arm in ARMS
        },
        "generator_training_eligible": HOLD,
        "runtime": {
            "device": str(device),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "batch_size": args.batch_size,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "limitations": [
            "300 additional epochs after10 residual epochs, not the entire experimental pipeline.",
            "Validation was inspected during earlier development; not an untouched test set.",
            "No test access. Absent/invalid-box frames are outside this eligible-pair audit.",
            "Same-index pairing; no new lag search or frame-pair repair.",
            "Hashes bind decoded inputs, not entire native video files.",
            "Same initialization/loss/update budget, intentionally different training samples.",
            "Box/cycle/topology checks do not establish independent physical correspondence.",
        ],
    }
    atomic_json(args.out_dir / "report.json", report)
    print(json.dumps({"report": str(args.out_dir / "report.json"), "gates": report["gates"]}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=antiuav300_root()
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("experiments/registration_replay_e300_seed0")
    )
    parser.add_argument(
        "--base-root", type=Path, default=Path("experiments/registration_residual_pilot_e10_seed0")
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-sequence", type=int, choices=(0, 8), default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
