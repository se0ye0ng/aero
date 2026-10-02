#!/usr/bin/env python3
"""Audit saved matches' ROI spread and box-transfer proxies without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.detector_free import inside_box, reciprocal_mask
from aero_ir.utils.manifest import file_sha256
from aero_ir.utils.paths import antiuav300_root


def hull_fraction(points, box):
    unique = np.unique(np.round(points, 1), axis=0)
    area = float(np.prod(box[2:] - box[:2]))
    if len(unique) < 3 or area <= 0:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(unique.astype(np.float32))) / area)


def transfer_iou(matrix, source_box, target_box):
    if matrix is None:
        return None
    matrix = np.asarray(matrix)
    x0, y0, x1, y1 = source_box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    homogeneous = np.c_[corners, np.ones(4)] @ matrix.T
    denominators = homogeneous[:, 2]
    # A homography pole crossing the box invalidates its finite enclosure.
    if (np.abs(denominators) < 1e-8).any() or not (
        (denominators > 0).all() or (denominators < 0).all()
    ):
        return None
    warped = homogeneous[:, :2] / denominators[:, None]
    if not np.isfinite(warped).all():
        return None
    low, high = warped.min(0), warped.max(0)
    intersection = np.maximum(
        np.minimum(high, target_box[2:]) - np.maximum(low, target_box[:2]), 0
    ).prod()
    union = (high - low).prod() + (target_box[2:] - target_box[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=antiuav300_root()
    )
    args = parser.parse_args()
    destination = args.report.parent / "spatial_audit.json"
    if destination.exists():
        parser.error("refusing to overwrite spatial audit")
    report = json.loads(args.report.read_text())
    if report["evaluated_split"] != "train" or report["validation_or_test_access"] != "none":
        raise ValueError("requires a train-only matching screen")
    inputs = {r["sequence_id"]: r for r in report["inputs"]}
    rows = []
    for row in report["rows"]:
        info = inputs[row["sequence_id"]]
        boxes = []
        for modality in ("visible", "infrared"):
            entry = info["inputs"][modality]
            path = args.root / "train" / row["sequence_id"] / f"{modality}.json"
            if file_sha256(path) != entry["annotation_sha256"]:
                raise ValueError("annotation provenance mismatch")
            x, y, w, h = json.loads(path.read_text())["gt_rect"][row["frame_index"]]
            nh, nw = entry["native_shape"]
            rh, rw = entry["resized_shape"]
            boxes.append(
                np.array([x, y, x + w, y + h]) * np.array([rw / nw, rh / nh, rw / nw, rh / nh])
            )
        path = args.report.parent / row["matches_file"]
        if file_sha256(path) != row["matches_sha256"]:
            raise ValueError("saved matches changed")
        matches = np.load(path)
        a, b, ra, rb = [matches[k] for k in ("points0", "points1", "reverse0", "reverse1")]
        keep = inside_box(a, boxes[0]) & inside_box(b, boxes[1]) & reciprocal_mask(a, b, ra, rb)
        both_pairs = np.c_[a[keep], b[keep]]
        unique = len(np.unique(np.round(both_pairs), axis=0))
        rows.append(
            {
                "model": row["model"],
                "sequence_id": row["sequence_id"],
                "unique_reciprocal_both_box_pairs_1px": unique,
                "rgb_hull_fraction": hull_fraction(a[keep], boxes[0]),
                "ir_hull_fraction": hull_fraction(b[keep], boxes[1]),
                "non_header_homography_box_iou_not_pixel_accuracy": {
                    d: transfer_iou(
                        row[d]["header_screen"][
                            "non_header_homography_consistency_not_accuracy"
                        ].get("matrix"),
                        *box_order,
                    )
                    for d, box_order in (("rgb_to_ir", boxes), ("ir_to_rgb", boxes[::-1]))
                },
            }
        )
    summary = {}
    for name in ("xoftr", "loftr"):
        selected = [r for r in rows if r["model"] == name]
        values = [
            r["non_header_homography_box_iou_not_pixel_accuracy"]["rgb_to_ir"] for r in selected
        ]
        available = [v for v in values if v is not None]
        summary[name] = {
            "pairs": len(selected),
            "pairs_with_four_unique_reciprocal_both_box_pairs": sum(
                r["unique_reciprocal_both_box_pairs_1px"] >= 4 for r in selected
            ),
            "pairs_also_covering_ten_percent_of_both_boxes": sum(
                r["unique_reciprocal_both_box_pairs_1px"] >= 4
                and min(r["rgb_hull_fraction"], r["ir_hull_fraction"]) >= 0.1
                for r in selected
            ),
            "non_header_homography_rgb_to_ir_box_iou": {
                "available": len(available),
                "mean": float(np.mean(available)) if available else None,
                "median": float(np.median(available)) if available else None,
                "count_at_least_0_6": sum(v >= 0.6 for v in available),
            },
        }
    result = {
        "report_sha256": file_sha256(args.report),
        "rows": rows,
        "summary": summary,
        "scope": (
            "exploratory train-only spread/box proxies; 10% hull is not a qualification threshold"
        ),
        "script_sha256": file_sha256(__file__),
    }
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
