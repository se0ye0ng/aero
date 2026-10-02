"""Train-only repair candidate for v4's loss/qualification mismatch.

This is an experimental objective, not a qualified registration protocol. The
v4 coordinate adapter and qualification thresholds are retained. Cycle errors
are optimized in pixels using per-frame tail losses, rather than averaging
normalized displacement over an entire batch. Independent physical evidence
is still required before authorizing a paired-image generator.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import (
    box_points,
    cycle_at,
    cycle_field,
    from_superfusion,
    jacobian_determinant,
    map_points,
    pixel_norm,
    point_jacobian_determinant,
    sampling_map,
)
from aero_ir.registration.protocol_v4 import _direction_loss
from aero_ir.registration.qualification_v4 import THRESHOLDS

INITIAL_CHECKPOINT_SHA256 = "db75c7121a2e52cf1ba7a212bd8fc47d58d5d536e078d937f25ecae01af3b91d"
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
TRAINING_MARGIN = 0.5
MINIMUM_JACOBIAN = 0.2
LOSS_WEIGHTS = {
    "box_iou": 1.0,
    "centroid_fraction": 0.25,
    "log_area_ratio": 0.25,
    "structural_edge_ncc": 0.05,
    "flow_smoothness": 0.10,
    "cycle_global_tail": 0.25,
    "cycle_roi_tail": 0.25,
    "cycle_roi_max": 0.25,
    "jacobian_global_tail": 0.10,
    "jacobian_roi": 0.10,
    "support_global": 0.10,
    "support_roi": 0.10,
}


def _tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean of the largest fraction; empty support contributes a differentiable zero.

    The explicit support objective below prevents empty support being considered
    a successful registration. An empty cycle domain has no measurable cycle.
    """
    if not values.numel():
        return values.sum()
    count = max(1, math.ceil(values.numel() * fraction))
    return values.flatten().topk(count).values.mean()


def _support_excursion(points, height, width):
    limit = points.new_tensor([1 - 1 / width, 1 - 1 / height])
    return pixel_norm(F.relu(points.abs() - limit), height, width)


def _global_support_loss(first, second):
    middle = sampling_map(first).flatten(1, 2)
    returned = map_points(second, middle)
    excursion = torch.maximum(
        _support_excursion(middle, *second.shape[2:]),
        _support_excursion(returned, *first.shape[2:]),
    )
    # Qualification requires 90% support, not physically impossible 100%
    # overlap across different camera fields of view. Penalize only the least
    # displaced 90%; the loss becomes zero once this support quota is attained.
    count = math.ceil(THRESHOLDS["minimum_valid_fraction"] * excursion.shape[1])
    observed = excursion.topk(count, dim=1, largest=False).values
    return torch.stack([_tail_mean(row, 0.05) for row in observed]).mean()


def _direction_constraints(first, second, boxes):
    height, width = first.shape[2:]
    residual, valid = cycle_field(first, second)
    errors = pixel_norm(residual, height, width)
    points = box_points(boxes)
    roi_residual, roi_valid = cycle_at(first, second, points)
    roi_errors = pixel_norm(roi_residual, height, width)
    diagonal = torch.linalg.vector_norm(boxes[:, 2:] * boxes.new_tensor([width, height]), dim=1)
    p95_limit = torch.minimum(
        diagonal * THRESHOLDS["maximum_roi_cycle_p95_diagonal_fraction"],
        diagonal.new_full(diagonal.shape, THRESHOLDS["maximum_roi_cycle_p95_pixels"]),
    ).clamp_min(1e-6)
    max_limit = torch.minimum(
        diagonal * THRESHOLDS["maximum_roi_cycle_max_diagonal_fraction"],
        diagonal.new_full(diagonal.shape, THRESHOLDS["maximum_roi_cycle_max_pixels"]),
    ).clamp_min(1e-6)
    global_losses, roi_losses, roi_max_losses = [], [], []
    for index in range(first.shape[0]):
        global_losses.append(
            _tail_mean(
                F.relu(
                    errors[index][valid[index]]
                    / THRESHOLDS["maximum_global_cycle_p95_pixels"]
                    - TRAINING_MARGIN
                ),
                0.05,
            )
        )
        local = roi_errors[index][roi_valid[index]]
        roi_losses.append(_tail_mean(F.relu(local / p95_limit[index] - TRAINING_MARGIN), 0.05))
        roi_max_losses.append(
            _tail_mean(
                F.relu(local / max_limit[index] - TRAINING_MARGIN),
                1 / max(1, local.numel()),
            )
        )
    determinant = jacobian_determinant(first)
    fold = F.relu((MINIMUM_JACOBIAN - determinant) / MINIMUM_JACOBIAN)
    roi_determinant = point_jacobian_determinant(first, points)
    # ROI derivative stencils beyond the lattice have no valid measurement.
    # Penalize their required support explicitly instead of treating NaN as 0.
    roi_fold = F.relu((MINIMUM_JACOBIAN - roi_determinant) / MINIMUM_JACOBIAN)
    roi_fold = torch.where(torch.isfinite(roi_fold), roi_fold, torch.zeros_like(roi_fold))
    middle = map_points(first, points)
    returned = map_points(second, middle)
    roi_support = torch.maximum(
        _support_excursion(middle, *second.shape[2:]),
        _support_excursion(returned, height, width),
    )
    ex, ey = points.new_tensor([2 / width, 0]), points.new_tensor([0, 2 / height])
    for offset in (ex, -ex, ey, -ey):
        roi_support = torch.maximum(roi_support, _support_excursion(points + offset, height, width))
    values = {
        "cycle_global_tail": torch.stack(global_losses).mean(),
        "cycle_roi_tail": torch.stack(roi_losses).mean(),
        "cycle_roi_max": torch.stack(roi_max_losses).mean(),
        "jacobian_global_tail": torch.stack([_tail_mean(row, 0.01) for row in fold]).mean(),
        "jacobian_roi": roi_fold.amax(dim=1).mean(),
        "support_global": _global_support_loss(first, second),
        "support_roi": roi_support.amax(dim=1).mean(),
    }
    return values


def bidirectional_geometry_loss(visible, infrared, visible_boxes, infrared_boxes, forward, reverse):
    """Consume raw SuperFusion fields, converting exactly once as in v4."""
    forward, reverse = from_superfusion(forward), from_superfusion(reverse)
    f = _direction_loss(visible, infrared, visible_boxes, infrared_boxes, forward)
    r = _direction_loss(infrared, visible, infrared_boxes, visible_boxes, reverse)
    # Replace v4's mean fold/boundary/cycle terms, retaining image/box supervision.
    values = {key: (f[key] + r[key]) / 2 for key in f if key in LOSS_WEIGHTS}
    f = _direction_constraints(forward, reverse, infrared_boxes)
    r = _direction_constraints(reverse, forward, visible_boxes)
    values.update({key: (f[key] + r[key]) / 2 for key in f})
    total = sum(LOSS_WEIGHTS[key] * value for key, value in values.items())
    return total, {
        **{key: float(value.detach()) for key, value in values.items()},
        "total": float(total.detach()),
    }
