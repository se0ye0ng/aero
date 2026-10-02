"""Experimental v5 tail objective with per-pair frozen-v4 preservation penalties.

These are soft constraints, not a guarantee of qualification or physical alignment.
The teacher supplies loss ceilings against the actual annotated boxes; its field
is never used as physical correspondence ground truth. Coordinates and evaluation
thresholds remain those of v4.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import from_superfusion, map_boxes
from aero_ir.registration.protocol_v5 import bidirectional_geometry_loss as v5_loss

# Fixed before the pilot; fractions are in normalized box-loss units, not pixels.
PRESERVATION_TOLERANCES = {
    "box_iou": 0.01,
    "centroid_fraction": 0.01,
    "log_area_ratio": 0.02,
}
PRESERVATION_WEIGHTS = {"box_iou": 10.0, "centroid_fraction": 5.0, "log_area_ratio": 2.5}


def box_errors(source_boxes, target_boxes, raw_field):
    """Return differentiable errors for every pair, without a batch reduction."""
    predicted = map_boxes(target_boxes, from_superfusion(raw_field))
    low, high = predicted[:, :2] - predicted[:, 2:] / 2, predicted[:, :2] + predicted[:, 2:] / 2
    truth_low = source_boxes[:, :2] - source_boxes[:, 2:] / 2
    truth_high = source_boxes[:, :2] + source_boxes[:, 2:] / 2
    intersection = (
        (torch.minimum(high, truth_high) - torch.maximum(low, truth_low)).clamp_min(0).prod(dim=1)
    )
    area, truth_area = predicted[:, 2:].prod(dim=1), source_boxes[:, 2:].prod(dim=1)
    diagonal = torch.linalg.vector_norm(source_boxes[:, 2:], dim=1).clamp_min(1e-5)
    return {
        "box_iou": 1 - intersection / (area + truth_area - intersection).clamp_min(1e-8),
        "centroid_fraction": (
            torch.linalg.vector_norm(predicted[:, :2] - source_boxes[:, :2], dim=1) / diagonal
        ),
        "log_area_ratio": torch.log((area + 1e-8) / (truth_area + 1e-8)).abs(),
    }


def preservation_penalty(student, teacher):
    """Hinge each pair/direction before reduction so improvements cannot cancel harm."""
    penalties = {
        key: F.relu(student[key] - teacher[key].detach() - margin).mean()
        for key, margin in PRESERVATION_TOLERANCES.items()
    }
    total = sum(PRESERVATION_WEIGHTS[key] * value for key, value in penalties.items())
    return total, penalties


def bidirectional_geometry_loss(
    visible, infrared, visible_boxes, infrared_boxes, forward, reverse,
    teacher_forward, teacher_reverse,
):
    total, metrics = v5_loss(
        visible, infrared, visible_boxes, infrared_boxes, forward, reverse
    )
    student_directions, teacher_directions = [], []
    for source_boxes, target_boxes, student, teacher in (
        (visible_boxes, infrared_boxes, forward, teacher_forward),
        (infrared_boxes, visible_boxes, reverse, teacher_reverse),
    ):
        student_directions.append(box_errors(source_boxes, target_boxes, student))
        teacher_directions.append(box_errors(source_boxes, target_boxes, teacher.detach()))
    student = {key: torch.cat([row[key] for row in student_directions])
               for key in PRESERVATION_TOLERANCES}
    teacher = {key: torch.cat([row[key] for row in teacher_directions])
               for key in PRESERVATION_TOLERANCES}
    penalty, values = preservation_penalty(student, teacher)
    total = total + penalty
    for key, value in values.items():
        metrics[f"preserve_{key}"] = float(value.detach())
        metrics[f"teacher_{key}"] = float(teacher[key].mean().detach())
        metrics[f"regression_{key}"] = float((student[key] - teacher[key]).mean().detach())
        metrics[f"violation_fraction_{key}"] = float(
            (student[key] > teacher[key] + PRESERVATION_TOLERANCES[key]).float().mean().detach()
        )
    metrics["total"] = float(total.detach())
    return total, metrics
