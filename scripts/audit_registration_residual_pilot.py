"""Evaluate completed matched residual pilots with unchanged engineering criteria.

Both pilots must verify, and at least one must pass the frozen train-midpoint
screen, before reading validation pixels. Neither this screen nor exhaustive
box/cycle checks establish independent physical correspondence or generator GO.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch

from aero_ir.registration.geometry import CONVENTION
from aero_ir.registration.qualification_v4 import THRESHOLDS, direction_pass, gate_report
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_registration_v7 import (
    SPLITS,
    Coverage,
    _batches,
    _iter_pairs,
    build_panel,
    check,
    infer_rows,
    source_hashes,
    write_json,
)
from scripts.compare_registration_residual_pilot import compare
from scripts.train_registration_residual_pilot import ARMS, load_checkpoint


def candidates(comparison):
    """Select only from verified train summaries, never validation outcomes."""
    return [
        arm for arm in ARMS
        if comparison["arms"][arm]["final"]["joint_frame_pass_rate"]
        >= THRESHOLDS["minimum_joint_frame_pass_rate"]
        and comparison["arms"][arm]["final"]["sequence_macro_pass_rate"]
        >= THRESHOLDS["minimum_sequence_macro_pass_rate"]
    ]


def pilot_hashes(root):
    """Bind evidence consumed by compare(), including all retained attempt logs."""
    result = {}
    for arm in ARMS:
        folder = root / arm
        files = [folder / name for name in (
            "run_spec.json", "training_result.json", "final.pth",
            "initial_train_screen.json", "final_train_screen.json",
        )]
        files += sorted(folder.glob("attempt_*.jsonl"))
        files += sorted(folder.glob("attempt_*_runtime.json"))
        result[arm] = {str(p.relative_to(folder)): file_sha256(p) for p in files}
    return result


def audit_sources():
    project = Path(__file__).resolve().parents[1]
    result = source_hashes(project)
    for name in (
        "scripts/audit_registration_residual_pilot.py",
        "scripts/train_registration_residual_pilot.py",
        "scripts/compare_registration_residual_pilot.py",
        "scripts/run_registration_residual_audit.sh",
    ):
        result[name] = file_sha256(project / name)
    return result


def paired_key(left, right):
    for key in ("split", "sequence_id", "frame_index", "array_sha256"):
        check(left[key] == right[key], "arm observations differ")
    flags = tuple(
        all(direction_pass(row[k]) for k in ("ir_to_rgb_points", "rgb_to_ir_points"))
        for row in (left, right)
    )
    return {
        (True, True): "both_pass", (False, True): "residual_only_pass",
        (True, False): "continuation_only_pass", (False, False): "neither_pass",
    }[flags]


def run(args):
    check(args.batch_size > 0, "batch size must be positive")
    check(args.samples_per_sequence in (0, 8), "use eight-frame or exhaustive eligible panel")
    for name in ("root", "run_root", "cache_root", "out_dir"):
        check(not any(c in str(getattr(args, name)) for c in ("\n", "\r")),
              f"newline in {name}")
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite audit directory; use a fresh --out-dir")
    sources = audit_sources()
    comparison, identities, eligible, error = None, None, [], None
    try:
        identities = pilot_hashes(args.run_root)
        comparison = compare(args.run_root, args.cache_root)
        check(pilot_hashes(args.run_root) == identities, "pilot changed during verification")
        eligible = candidates(comparison)
    except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    ready = bool(eligible) and error is None
    plan = {
        "schema_version": 1, "kind": "residual_pilot_inference_audit_preflight",
        "ready_for_engineering_inference": ready,
        "candidates_from_train_screen": eligible,
        "comparison": comparison, "pilot_artifacts_sha256": identities,
        "blocked_reason": None if ready else error or
            "Neither completed arm reached both frozen 95% train-screen pass rates.",
        "source_sha256": sources, "coordinate_convention": CONVENTION,
        "thresholds": THRESHOLDS, "pixel_inference_executed": False,
        "generator_training_eligible": "hold_not_qualified",
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if not ready:
        write_json(args.out_dir / "preflight.json", plan)
        print(json.dumps({"ready": False, "reason": plan["blocked_reason"]}))
        return 2
    # Only now select validation observations. No frames are decoded by build_panel.
    expected, panel = build_panel(args.root, args.cache_root, args.samples_per_sequence)
    panel["kind"] = "residual_pilot_frozen_annotation_selected_evaluation_panel"
    plan["panel_sha256"] = canonical_hash(panel)
    plan["cache_manifest_sha256"] = file_sha256(args.cache_root / "manifest.json")
    plan["selected_pairs"] = {s: len(v) for s, v in expected.items()}
    write_json(args.out_dir / "panel.json", panel)
    write_json(args.out_dir / "preflight.json", plan)
    if args.preflight_only:
        print(json.dumps({"ready": True, "selected_pairs": plan["selected_pairs"]}))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; preflight preserved, no inference started")
    models = {}
    for arm in ARMS:
        checkpoint = args.run_root / arm / "final.pth"
        check(file_sha256(checkpoint) == identities[arm]["final.pth"], "checkpoint changed")
        models[arm], state = load_checkpoint(checkpoint, device)
        check(state["arm"] == arm, "checkpoint arm differs from audit arm")
    coverage = {a: Coverage(expected) for a in ARMS}
    outcomes = {s: dict.fromkeys(
        ("both_pass", "residual_only_pass", "continuation_only_pass", "neither_pass"), 0
    ) for s in SPLITS}
    with ExitStack() as stack:
        outputs = {a: stack.enter_context((args.out_dir / f"{a}.jsonl").open("x")) for a in ARMS}
        for batch in _batches(
            _iter_pairs(args.root, SPLITS, args.samples_per_sequence), args.batch_size
        ):
            predictions = {a: infer_rows(models[a], batch, device) for a in ARMS}
            for arm, rows in predictions.items():
                for row in rows:
                    coverage[arm].add(row)
                    outputs[arm].write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                outputs[arm].flush()
            for left, right in zip(predictions[ARMS[0]], predictions[ARMS[1]], strict=True):
                outcomes[left["split"]][paired_key(left, right)] += 1
            print(f"Audited {sum(len(v) for v in coverage[ARMS[0]].seen.values())} pairs",
                  flush=True)
    check(audit_sources() == sources, "audit source drift during inference")
    check(pilot_hashes(args.run_root) == identities, "pilot artifact drift during inference")
    check(file_sha256(args.cache_root / "manifest.json") == plan["cache_manifest_sha256"],
          "cache manifest drift during inference")
    for name, digest in panel["annotation_sha256"].items():
        check(file_sha256(args.root / name) == digest, "annotation drift during inference")
    metrics = {a: coverage[a].finish() for a in ARMS}
    signatures = {a: coverage[a].data_signature.hexdigest() for a in ARMS}
    check(len(set(signatures.values())) == 1, "different consumed evaluation observations")
    complete = not any(panel["sequences_without_eligible_pairs"].values())
    report = {
        "schema_version": 1, "kind": "matched_residual_pilot_engineering_inference_audit",
        "panel_sha256": file_sha256(args.out_dir / "panel.json"),
        "preflight_sha256": file_sha256(args.out_dir / "preflight.json"),
        "row_artifacts_sha256": {a: file_sha256(args.out_dir / f"{a}.jsonl") for a in ARMS},
        "source_sha256": sources, "coordinate_convention": CONVENTION,
        "field_conversion": "direct centre fields from model.fields; no raw-field adapter",
        "thresholds": THRESHOLDS, "metrics": metrics, "paired_outcomes": outcomes,
        "consumed_pair_signatures": signatures, "exact_eligible_pair_coverage": True,
        "all_official_sequences_have_observations": complete,
        "gates": {a: gate_report(metrics[a], exhaustive=args.samples_per_sequence == 0,
                                 complete_coverage=complete) for a in ARMS},
        "generator_training_eligible": "hold_not_qualified",
        "runtime": {"device": str(device), "torch": str(torch.__version__),
                    "cuda": torch.version.cuda, "batch_size": args.batch_size,
                    "device_name": torch.cuda.get_device_name(device)
                    if device.type == "cuda" else "CPU"},
        "limitations": [
            "Ten additional pilot epochs, not the final300-epoch experiment.",
            "Validation was inspected in earlier development; not an untouched test set.",
            "No test access. Absent/invalid-box frames are outside this eligible-pair audit.",
            "Same-index pairing; no new lag search or frame-pair repair.",
            "Hashes bind decoded inputs, not complete video files.",
            "Equal budget, different learning rates/trainable capacity; not architecture-only.",
            "Box/cycle/topology checks do not establish independent physical correspondence.",
        ],
    }
    write_json(args.out_dir / "report.json", report)
    print(json.dumps({"report": str(args.out_dir / "report.json"), "gates": report["gates"]}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/lustre/winston1214/dataset/Anti-UAV300"))
    parser.add_argument("--run-root", type=Path,
                        default=Path("experiments/registration_residual_pilot_e10_seed0"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-sequence", type=int, choices=(0, 8), default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
