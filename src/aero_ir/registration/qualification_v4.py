"""Fail-closed engineering checks, explicitly separate from scientific qualification.

Thresholds are prospective engineering choices, not a claim of validated sensor
accuracy. Independent physical correspondences are deliberately a separate HOLD;
no CLI flag can turn box agreement into dense-registration evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from aero_ir.registration.geometry import (
    box_points,
    cycle_at,
    cycle_field,
    jacobian_determinant,
    map_boxes,
    map_points,
    pixel_norm,
    point_jacobian_determinant,
    sampling_map,
    valid_support,
)

THRESHOLDS = {
    "minimum_bbox_iou": 0.6,
    "maximum_centroid_shift_fraction": 0.25,
    "maximum_absolute_area_ratio_change": 0.5,
    "minimum_joint_frame_pass_rate": 0.95,
    "minimum_sequence_macro_pass_rate": 0.95,
    "minimum_valid_fraction": 0.90,
    "minimum_global_positive_jacobian_fraction": 0.99,
    "minimum_roi_valid_fraction": 1.0,
    "minimum_roi_positive_jacobian_fraction": 1.0,
    "maximum_global_cycle_p95_pixels": 1.0,
    "maximum_roi_cycle_p95_pixels": 1.0,
    "maximum_roi_cycle_max_pixels": 2.0,
    "maximum_roi_cycle_p95_diagonal_fraction": 0.10,
    "maximum_roi_cycle_max_diagonal_fraction": 0.25,
}


def _box_statistics(predicted, truth):
    low, high = predicted[:2] - predicted[2:] / 2, predicted[:2] + predicted[2:] / 2
    truth_low, truth_high = truth[:2] - truth[2:] / 2, truth[:2] + truth[2:] / 2
    overlap = (torch.minimum(high, truth_high) - torch.maximum(low, truth_low)).clamp_min(0).prod()
    a, b = predicted[2:].prod(), truth[2:].prod()
    return {
        "bbox_iou": float(overlap / (a + b - overlap).clamp_min(1e-12)),
        "centroid_shift_fraction": float(
            torch.linalg.vector_norm(predicted[:2] - truth[:2])
            / torch.linalg.vector_norm(truth[2:]).clamp_min(1e-12)
        ),
        "absolute_area_ratio_change": float((a / b.clamp_min(1e-12) - 1).abs()),
        "box_in_bounds": bool((low >= 0).all() and (high <= 1).all()),
    }


def _quantile(values: torch.Tensor, q: float) -> float | None:
    if not values.numel() or not torch.isfinite(values).all():
        return None
    return float(torch.quantile(values, q))


def direction_statistics(first, second, target_boxes, source_boxes) -> list[dict]:
    """Fields already use centre coordinates. Target is the first field's output domain."""
    height, width = first.shape[2:]
    source_height, source_width = second.shape[2:]
    residual, valid = cycle_field(first, second)
    norms = pixel_norm(residual, height, width)
    points = box_points(target_boxes)
    roi_residual, roi_valid = cycle_at(first, second, points)
    roi_norms = pixel_norm(roi_residual, height, width)
    roi_jacobian = point_jacobian_determinant(first, points)
    determinants = jacobian_determinant(first)
    mapped_boxes = map_boxes(target_boxes, first)
    support = valid_support(sampling_map(first), source_height, source_width)
    point_support = valid_support(map_points(first, points), source_height, source_width)
    diagonal = torch.linalg.vector_norm(
        target_boxes[:, 2:] * first.new_tensor([width, height]), dim=1
    )
    rows = []
    for i in range(first.shape[0]):
        rows.append(
            {
                **_box_statistics(mapped_boxes[i], source_boxes[i]),
                "finite_field": bool(
                    torch.isfinite(first[i]).all() and torch.isfinite(second[i]).all()
                ),
                "valid_fraction": float(support[i].float().mean()),
                "positive_jacobian_fraction": float((determinants[i] > 0).float().mean()),
                "cycle_valid_fraction": float(valid[i].float().mean()),
                "cycle_p95_pixels": _quantile(norms[i][valid[i]], 0.95),
                "roi_valid_fraction": float((roi_valid[i] & point_support[i]).float().mean()),
                "roi_positive_jacobian_fraction": float((roi_jacobian[i] > 0).float().mean()),
                "roi_cycle_p95_pixels": _quantile(roi_norms[i][roi_valid[i]], 0.95),
                "roi_cycle_max_pixels": _quantile(roi_norms[i][roi_valid[i]], 1.0),
                "roi_diagonal_pixels": float(diagonal[i]),
            }
        )
    # Keep failure reports standards-compliant JSON; None is a failing measurement.
    return [
        {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def direction_pass(row: dict) -> bool:
    """Missing/non-finite values and zero observed support fail, never silently pass."""

    def at_least(key, threshold):
        value = row.get(key)
        return isinstance(value, (float, int)) and math.isfinite(value) and value >= threshold

    def at_most(key, threshold):
        value = row.get(key)
        return isinstance(value, (float, int)) and math.isfinite(value) and value <= threshold

    t = THRESHOLDS
    diagonal = row.get("roi_diagonal_pixels")
    if not isinstance(diagonal, (float, int)) or not math.isfinite(diagonal) or diagonal <= 0:
        return False
    return bool(
        row.get("finite_field") is True
        and row.get("box_in_bounds") is True
        and at_least("bbox_iou", t["minimum_bbox_iou"])
        and at_most("centroid_shift_fraction", t["maximum_centroid_shift_fraction"])
        and at_most("absolute_area_ratio_change", t["maximum_absolute_area_ratio_change"])
        and at_least("valid_fraction", t["minimum_valid_fraction"])
        and at_least("cycle_valid_fraction", t["minimum_valid_fraction"])
        and at_least("positive_jacobian_fraction", t["minimum_global_positive_jacobian_fraction"])
        and at_least("roi_valid_fraction", t["minimum_roi_valid_fraction"])
        and at_least("roi_positive_jacobian_fraction", t["minimum_roi_positive_jacobian_fraction"])
        and at_most("cycle_p95_pixels", t["maximum_global_cycle_p95_pixels"])
        and at_most(
            "roi_cycle_p95_pixels",
            min(
                t["maximum_roi_cycle_p95_pixels"],
                t["maximum_roi_cycle_p95_diagonal_fraction"] * diagonal,
            ),
        )
        and at_most(
            "roi_cycle_max_pixels",
            min(
                t["maximum_roi_cycle_max_pixels"],
                t["maximum_roi_cycle_max_diagonal_fraction"] * diagonal,
            ),
        )
    )


def split_report(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no v4 registration pairs evaluated")
    sequences = defaultdict(list)
    all_pass = []
    for row in rows:
        # Require all conditions and both directions on the SAME frame.
        passed = direction_pass(row["ir_to_rgb_points"]) and direction_pass(row["rgb_to_ir_points"])
        all_pass.append(passed)
        sequences[row["sequence_id"]].append(passed)
    per_sequence = {key: sum(values) / len(values) for key, values in sorted(sequences.items())}
    return {
        "evaluated_pairs": len(rows),
        "sequences": len(sequences),
        "joint_frame_pass_rate": sum(all_pass) / len(all_pass),
        "sequence_macro_pass_rate": sum(per_sequence.values()) / len(per_sequence),
        "per_sequence_pass_rate": per_sequence,
    }


def gate_report(metrics: dict, *, exhaustive: bool, complete_coverage: bool) -> dict:
    passed = complete_coverage and set(metrics) == {"train", "val"}
    for split in metrics.values():
        passed = passed and (
            split["joint_frame_pass_rate"] >= THRESHOLDS["minimum_joint_frame_pass_rate"]
            and split["sequence_macro_pass_rate"] >= THRESHOLDS["minimum_sequence_macro_pass_rate"]
        )
    return {
        "geometric_screen": "pass" if passed else "hold",
        "exhaustive_geometry": "pass" if passed and exhaustive else "hold",
        "independent_correspondence": "hold_not_evaluated",
        "dense_paired_image_registration": "hold",
        "generator_training_eligible": "hold",
    }
