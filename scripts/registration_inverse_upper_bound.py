#!/usr/bin/env python3
"""Bound joint pass when ONE native map is frozen and only its reciprocal changes.

All counted constraints depend solely on the frozen map. Passing them is necessary,
not sufficient. No bound is asserted for interventions that modify both maps.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from aero_ir.utils.manifest import canonical_hash, file_sha256


def bound(rows, direction, thresholds):
    sequences = defaultdict(list)
    failures = defaultdict(int)
    for row in rows:
        values = row[direction]
        checks = {"box_in_bounds": values["box_in_bounds"] is True}
        for key, threshold, lower in (
            ("bbox_iou", "minimum_bbox_iou", True),
            ("centroid_shift_fraction", "maximum_centroid_shift_fraction", False),
            ("absolute_area_ratio_change", "maximum_absolute_area_ratio_change", False),
            ("valid_fraction", "minimum_valid_fraction", True),
            ("positive_jacobian_fraction", "minimum_global_positive_jacobian_fraction", True),
            ("roi_positive_jacobian_fraction", "minimum_roi_positive_jacobian_fraction", True),
        ):
            v = values[key]
            checks[key] = (
                v is not None
                and math.isfinite(v)
                and (v >= thresholds[threshold] if lower else v <= thresholds[threshold])
            )
        sequences[row["sequence_id"]].append(all(checks.values()))
        for key, passed in checks.items():
            failures[key] += not passed
    upper = sum(sum(items) for items in sequences.values()) / len(rows)
    macro = sum(sum(items) / len(items) for items in sequences.values()) / len(sequences)
    return {
        "frames": len(rows),
        "joint_frame_pass_upper_bound": upper,
        "sequence_macro_pass_upper_bound": macro,
        "criterion_failure_counts_nonexclusive": dict(failures),
        "can_reach_frozen_threshold": (
            upper >= thresholds["minimum_joint_frame_pass_rate"]
            and macro >= thresholds["minimum_sequence_macro_pass_rate"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite a previous bound")
    report = json.loads(args.report.read_text())
    signature = report.pop("report_sha256")
    if canonical_hash(report) != signature:
        raise ValueError("comparison report hash mismatch")
    results = {
        name: {
            split: {
                direction: bound(rows, direction, report["thresholds"])
                for direction in ("ir_to_rgb_points", "rgb_to_ir_points")
            }
            for split, rows in item["rows"].items()
        }
        for name, item in report["models"].items()
    }
    output = {
        "kind": "fixed_native_map_joint_pass_upper_bound",
        "input_report_sha256": file_sha256(args.report),
        "models": results,
        "scope": "any inverse-only intervention preserving that native map exactly",
        "not_applicable_to": "training or algorithms modifying the preserved map",
    }
    output["report_sha256"] = canonical_hash(output)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    print(json.dumps({name: splits["val"] for name, splits in results.items()}, indent=2))


if __name__ == "__main__":
    main()
