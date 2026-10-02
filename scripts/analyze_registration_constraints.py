"""Explain existing registration measurements without fitting or changing gates.

Counts are engineering-proxy failures, not causal estimates or correspondence GT.
Counterfactual rates assume selected constraints become perfect and all others
stay fixed; they are ceilings for that hypothetical, never achieved results.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from aero_ir.registration.qualification_v4 import THRESHOLDS, direction_pass, split_report
from aero_ir.utils.manifest import file_sha256

DIRECTIONS = ("ir_to_rgb_points", "rgb_to_ir_points")
GROUPS = {
    "alignment": ("bbox_iou", "centroid_shift_fraction", "absolute_area_ratio_change"),
    "support": ("box_in_bounds", "valid_fraction", "cycle_valid_fraction", "roi_valid_fraction"),
    "topology": ("positive_jacobian_fraction", "roi_positive_jacobian_fraction"),
    "cycle": ("cycle_p95_pixels", "roi_cycle_p95_pixels", "roi_cycle_max_pixels"),
    "measurement_validity": ("finite_field", "roi_diagonal_pixels"),
}


def failures(row):
    """Mirror each frozen gate predicate, then assert agreement with the gate."""
    t = THRESHOLDS

    def number(value):
        # Match the frozen gate's numeric predicate, including its exact bounds.
        return isinstance(value, (float, int)) and math.isfinite(value)

    checks = {k: row.get(k) is True for k in ("finite_field", "box_in_bounds")}
    diagonal = row.get("roi_diagonal_pixels")
    checks["roi_diagonal_pixels"] = number(diagonal) and diagonal > 0
    lower = {
        "bbox_iou": t["minimum_bbox_iou"],
        "valid_fraction": t["minimum_valid_fraction"],
        "cycle_valid_fraction": t["minimum_valid_fraction"],
        "positive_jacobian_fraction": t["minimum_global_positive_jacobian_fraction"],
        "roi_valid_fraction": t["minimum_roi_valid_fraction"],
        "roi_positive_jacobian_fraction": t["minimum_roi_positive_jacobian_fraction"],
    }
    upper = {
        "centroid_shift_fraction": t["maximum_centroid_shift_fraction"],
        "absolute_area_ratio_change": t["maximum_absolute_area_ratio_change"],
        "cycle_p95_pixels": t["maximum_global_cycle_p95_pixels"],
    }
    if checks["roi_diagonal_pixels"]:
        for name, bound, fraction in (
            (
                "roi_cycle_p95_pixels",
                "maximum_roi_cycle_p95_pixels",
                "maximum_roi_cycle_p95_diagonal_fraction",
            ),
            (
                "roi_cycle_max_pixels",
                "maximum_roi_cycle_max_pixels",
                "maximum_roi_cycle_max_diagonal_fraction",
            ),
        ):
            upper[name] = min(t[bound], t[fraction] * diagonal)
    else:
        checks["roi_cycle_p95_pixels"] = checks["roi_cycle_max_pixels"] = False
    for key, limit in lower.items():
        checks[key] = number(row.get(key)) and row[key] >= limit
    for key, limit in upper.items():
        checks[key] = number(row.get(key)) and row[key] <= limit
    failed = {key for key, passed in checks.items() if not passed}
    if (not failed) != direction_pass(row):
        raise ValueError("constraint decomposition disagrees with the frozen gate")
    return failed


def analyze(rows):
    if not rows:
        raise ValueError("empty measurements")
    identities = [(r["sequence_id"], r.get("frame_index")) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("duplicate observations (multi-frame rows need frame_index)")
    summary = split_report(rows)
    by_direction = {d: Counter() for d in DIRECTIONS}
    joint = Counter()
    details = []
    sets = []
    for row in rows:
        item = {d: failures(row[d]) for d in DIRECTIONS}
        for d, failed in item.items():
            by_direction[d].update(failed)
        failed = set.union(*item.values())
        sets.append(failed)
        joint.update(failed)
        details.append(
            {
                "sequence_id": row["sequence_id"],
                "frame_index": row.get("frame_index"),
                "failed_constraints": sorted(failed),
                "by_direction": {d: sorted(v) for d, v in item.items()},
            }
        )
    hypothetical = {}
    for group, keys in GROUPS.items():
        passed = [not (failed - set(keys)) for failed in sets]
        sequences = defaultdict(list)
        for row, value in zip(rows, passed, strict=True):
            sequences[row["sequence_id"]].append(value)
        hypothetical[group] = {
            "joint_frame_pass_rate": sum(passed) / len(passed),
            "sequence_macro_pass_rate": sum(sum(v) / len(v) for v in sequences.values())
            / len(sequences),
            "newly_passing_frames_if_only_this_group_perfect": sum(
                bool(failed) and good for failed, good in zip(sets, passed, strict=True)
            ),
        }
    passed_count = sum(not s for s in sets)
    return {
        "observed": summary,
        "joint_failed_frame_count": len(rows) - passed_count,
        "minimum_frame_repairs_for_joint_threshold_only": max(
            0,
            math.ceil(len(rows) * THRESHOLDS["minimum_joint_frame_pass_rate"]) - passed_count,
        ),
        "failed_frames_by_constraint_overlapping": dict(sorted(joint.items())),
        "failed_frames_by_direction": {d: dict(sorted(c.items())) for d, c in by_direction.items()},
        "hypothetical_perfect_group_not_an_achieved_result": hypothetical,
        "rows": details,
    }


def target_scale_summary(rows, samples, cache):
    """Post-hoc size strata on the exact saved observations, never a new gate."""
    from scripts.train_antiuav300_registration_v7 import read_batch

    if not isinstance(samples, list) or len(samples) != len(rows):
        raise ValueError("target-scale analysis requires one saved sample per screen row")
    groups = {key: [] for key in ("below_4", "4_to_below_8", "8_to_below_16", "at_least_16")}
    observations = []
    for row, sample in zip(rows, samples, strict=True):
        selected = sample["selection"]
        if len(selected) != 1 or selected[0][0] != row["sequence_id"]:
            raise ValueError("sample selection differs from the screen row")
        arrays, digest = read_batch(cache, selected)
        if digest != sample["array_sha256"]:
            raise ValueError("target-scale observations differ from the saved screen")
        sides = []
        for image_key, box_key in (("visible", "source_boxes"), ("infrared", "target_boxes")):
            height, width = arrays[image_key].shape[1:3]
            box = arrays[box_key][0]
            sides.append(min(float(box[2]) * width, float(box[3]) * height))
        short_side = min(sides)
        if not math.isfinite(short_side) or short_side <= 0:
            raise ValueError("invalid target size")
        group = (
            "below_4"
            if short_side < 4
            else "4_to_below_8"
            if short_side < 8
            else "8_to_below_16"
            if short_side < 16
            else "at_least_16"
        )
        passed = all(direction_pass(row[d]) for d in DIRECTIONS)
        groups[group].append(passed)
        observations.append(
            {
                "sequence_id": row["sequence_id"],
                "selection": selected,
                "array_sha256": digest,
                "short_sides_visible_ir_pixels": sides,
                "stratum": group,
                "joint_pass": passed,
            }
        )
    return {
        "definition": "minimum annotated short side across visible and IR cached input grids",
        "units": "cached network-input pixels, not native camera pixels",
        "scope": "post_hoc_train_diagnostic_not_causal_evidence_or_qualification",
        "strata": {
            key: {
                "frames": len(v),
                "passed": sum(v),
                "joint_frame_pass_rate": sum(v) / len(v) if v else None,
            }
            for key, v in groups.items()
        },
        "observations": observations,
    }


def run(input_path, output_path, cache=None):
    if output_path.exists():
        raise FileExistsError(output_path)
    before = file_sha256(input_path)
    if input_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
        original = {}
        provenance = {"scope": "supplied streamed measurements; completion not established"}
    else:
        original = json.loads(input_path.read_text())
        rows = original["rows"]
        provenance = {
            k: original[k]
            for k in (
                "evaluated_split",
                "not_independent_of_training",
                "validation_or_test_access",
                "checkpoint_sha256",
                "spec_sha256",
            )
            if k in original
        }
    if file_sha256(input_path) != before:
        raise ValueError("input changed while reading; wait for the report to finish")
    if not rows:
        raise ValueError("empty measurements")
    by_split = defaultdict(list)
    for row in rows:
        split = row.get("split", original.get("evaluated_split", "unspecified"))
        by_split[split].append(row)
    results = {split: analyze(items) for split, items in sorted(by_split.items())}
    if "summary" in original:
        if len(results) != 1 or original["summary"] != next(iter(results.values()))["observed"]:
            raise ValueError("saved summary differs from recomputed per-frame measurements")
    project = Path(__file__).resolve().parents[1]
    result = {
        "kind": "registration_constraint_attribution_not_qualification",
        "input": {"path": str(input_path.resolve()), "sha256": before},
        "input_provenance": provenance,
        "thresholds_unchanged": THRESHOLDS,
        "source_sha256": {
            name: file_sha256(project / name)
            for name in (
                "scripts/analyze_registration_constraints.py",
                "src/aero_ir/registration/qualification_v4.py",
            )
        },
        "splits": results,
        "limits": [
            "No inference replay, full coverage verification, threshold tuning or fitting.",
            "Overlapping failures are not additive and are not causal diagnoses.",
            "Perfect-group hypotheticals can include physically unrepairable missing support.",
            "Read initial/final/checkpoint scope from the input; "
            "this report cannot prove completion.",
            "Frame repair count does not by itself ensure the sequence-macro threshold.",
        ],
        "generator_training_eligible": "hold_not_qualified",
    }
    if cache is not None:
        if original.get("evaluated_split") != "train":
            raise ValueError("target-scale analysis only accepts saved train screens")
        result["target_scale"] = target_scale_summary(rows, original.get("samples"), cache)
        result["source_sha256"]["scripts/train_antiuav300_registration_v7.py"] = file_sha256(
            project / "scripts/train_antiuav300_registration_v7.py"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="optional v7 cache for hash-checked train target-size strata",
    )
    args = parser.parse_args()
    result = run(args.input, args.out, args.cache_root)
    for split, values in result["splits"].items():
        print(split, json.dumps(values["failed_frames_by_constraint_overlapping"]))
    print(f"wrote {args.out}; diagnostic only, no qualification approval")


if __name__ == "__main__":
    main()
