"""Export aggregate verified results, never private paths/images or a GO claim.

Read-only checks of existing FLIR/residual products precede export. No inference,
training, new dataset evaluation, git operation or external publication occurs.
"""

import argparse
import json
from pathlib import Path

from aero_ir.registration.qualification_v4 import THRESHOLDS
from aero_ir.utils.manifest import file_sha256, load_manifest, verify_run_manifest
from scripts.compare_registration_residual_pilot import compare

METRICS = ("map_50", "map_50_95", "mar_50", "mar_50_95")


def aggregate(manifest, baseline_proof, registration):
    if not baseline_proof["ok"] or not registration["matched_budget_and_data"]:
        raise ValueError("unverified inputs")
    metrics = manifest["metrics"]
    arms = {}
    for arm in ("predictor_continuation", "residual_head"):
        record = registration["arms"][arm]
        final = record["final"]
        rates = final["per_sequence_pass_rate"]
        if len(rates) != final["evaluated_pairs"] or any(v not in (0, 1) for v in rates.values()):
            raise ValueError("expected exactly one midpoint per train sequence")
        arms[arm] = {
            "epochs": 10,
            "optimizer_steps": record["trace"]["verified_optimizer_steps"],
            "joint_passes": sum(int(v) for v in rates.values()),
            "evaluated_pairs": final["evaluated_pairs"],
            "joint_pass_rate": final["joint_frame_pass_rate"],
            "learning_rate": registration["learning_rates"][arm],
            "checkpoint_sha256": record["final_checkpoint_sha256"],
        }
    return {
        "schema_version": 1,
        "kind": "aero_public_aggregate_results",
        "scope": "verified_saved_artifacts_not_new_training_or_inference",
        "flir_real_only": {
            "run_id": manifest["run_id"],
            "split": "validation",
            "epochs": 300,
            "images": metrics["images"],
            "detections": metrics["detections"],
            "metrics_fraction": {k: metrics[k] for k in METRICS},
            "per_class_fraction": metrics["per_class"],
            "manifest_sha256": baseline_proof["manifest_sha256"],
            "original_source_commit": manifest["git_sha"],
            "input_sources_verified_from_original_git": baseline_proof["verified_from_git"],
            "replayed_during_this_export": baseline_proof["replayed"],
        },
        "registration_pilots": {
            "dataset": "Anti-UAV300",
            "split": "train",
            "selection": "one midpoint per sequence",
            "initial_joint_pass_rate": registration["arms"]["residual_head"]["initial"][
                "joint_frame_pass_rate"
            ],
            "arms": arms,
            "matched_training_data_and_update_count": True,
            "architecture_and_learning_rate_both_differ": True,
            "minimum_joint_pass_rate": THRESHOLDS["minimum_joint_frame_pass_rate"],
            "independent_physical_accuracy_established": False,
        },
        "three_arm_detector_comparison": {
            "real_plus_simulated": "not_completed",
            "real_plus_generated": "not_completed",
            "generated_data_ap_improvement": None,
        },
        "registration_qualification": "hold_not_qualified",
        "central_hypothesis_status": "not_tested_by_a_completed_three_arm_experiment",
        "limitations": [
            "Detection AP and registration pass rates are different metrics on different datasets.",
            "Registration pilots are train diagnostics, not held-out physical accuracy.",
            "Artifact verification does not replay GPU inference or training.",
            "No completed300-additional-epoch replay result is included in this frozen snapshot.",
            "Source hashes are provenance, not a substitute for access to original artifacts.",
        ],
    }


def build(project):
    baseline_path = (
        project / "experiments/yolox_runs/flir_real_only_full_seed0_v2/run_manifest.json"
    )
    residual_root = project / "experiments/registration_residual_pilot_e10_seed0"
    proof = verify_run_manifest(baseline_path, execute=False)
    manifest = load_manifest(baseline_path)
    if manifest["manifest_sha256"] != proof["manifest_sha256"]:
        raise ValueError("baseline manifest changed during verification")
    registration = compare(
        residual_root, project / "experiments/antiuav300_registration_v2_full_train_cache"
    )
    recorded = json.loads((residual_root / "comparison.json").read_text())
    if recorded != {
        **registration,
        "comparison_source_sha256": file_sha256(
            project / "scripts/compare_registration_residual_pilot.py"
        ),
    }:
        raise ValueError("saved registration comparison differs from recomputed evidence")
    result = aggregate(manifest, proof, registration)
    names = [baseline_path, residual_root / "comparison.json"]
    for arm in registration["arms"]:
        names += [
            residual_root / arm / "run_spec.json",
            residual_root / arm / "final_train_screen.json",
        ]
    result["source_artifacts_sha256"] = {str(p.relative_to(project)): file_sha256(p) for p in names}
    result["exporter_source_sha256"] = file_sha256(__file__)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.out and args.check:
        parser.error("choose export or read-only check")
    result = build(Path(__file__).resolve().parents[1])
    if args.check:
        if json.loads(args.check.read_text()) != result:
            raise ValueError("public result snapshot differs from verified local artifacts")
        print("Public result snapshot matches verified local artifacts; qualification remains HOLD")
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
