"""Coordinate-consistent v4 loss. Geometry checks alone never qualify paired data.

Inputs to the loss are RAW matcher fields; they are converted exactly once.
Box enclosures are weak supervision, not corresponding physical boundary points.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import (
    box_points,
    cycle_at,
    cycle_field,
    from_superfusion,
    jacobian_determinant,
    map_boxes,
    sampling_map,
    warp,
)
from aero_ir.registration.protocol_v2 import _edge_map

EPOCHS = 300
PAIRS_PER_SEQUENCE_PER_EPOCH = 16
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
INITIAL_CHECKPOINT_SHA256 = "13dd5c4700aa072e8b51197222d0d36675e4ab5a4879abd940576d579331135f"
LOSS_WEIGHTS = {
    "box_iou": 1.0,
    "centroid_fraction": 0.25,
    "log_area_ratio": 0.25,
    "structural_edge_ncc": 0.05,
    "flow_smoothness": 0.10,
    "flow_fold": 0.10,
    "flow_boundary": 0.10,
    "inverse_consistency": 0.25,
    "roi_inverse_consistency": 0.25,
}


def masked_edge_ncc(
    source: torch.Tensor, target: torch.Tensor, field: torch.Tensor
) -> torch.Tensor:
    """Structural diagnostic on observed overlap; not a ground-truth correspondence score."""
    a = warp(_edge_map(source), field)[:, 0]
    b = _edge_map(target)[:, 0]
    # Require a one-pixel margin for the Sobel stencil in both images.
    grid = sampling_map(field)
    h, w = source.shape[2:]
    margin = grid.new_tensor([1 - 3 / w, 1 - 3 / h])
    mask = (grid.abs() <= margin).all(dim=-1)
    mask[:, :1] = False
    mask[:, -1:] = False
    mask[:, :, :1] = False
    mask[:, :, -1:] = False
    count = mask.sum(dim=(1, 2)).clamp_min(1)
    a = (a - (a * mask).sum(dim=(1, 2))[:, None, None] / count[:, None, None]) * mask
    b = (b - (b * mask).sum(dim=(1, 2))[:, None, None] / count[:, None, None]) * mask
    numerator = (a * b).sum(dim=(1, 2))
    denominator = (
        a.square().sum(dim=(1, 2)).clamp_min(1e-12).sqrt()
        * b.square().sum(dim=(1, 2)).clamp_min(1e-12).sqrt()
    )
    return numerator / denominator.clamp_min(1e-12)


def _direction_loss(source, target, source_boxes, target_boxes, field):
    predicted = map_boxes(target_boxes, field)
    low = predicted[:, :2] - predicted[:, 2:] / 2
    high = predicted[:, :2] + predicted[:, 2:] / 2
    truth_low = source_boxes[:, :2] - source_boxes[:, 2:] / 2
    truth_high = source_boxes[:, :2] + source_boxes[:, 2:] / 2
    intersection = (torch.minimum(high, truth_high) - torch.maximum(low, truth_low)).clamp_min(0)
    intersection = intersection.prod(dim=1)
    area, truth_area = predicted[:, 2:].prod(dim=1), source_boxes[:, 2:].prod(dim=1)
    diagonal = torch.linalg.vector_norm(source_boxes[:, 2:], dim=1).clamp_min(1e-5)
    h, w = source.shape[2:]
    limit = field.new_tensor([1 - 1 / w, 1 - 1 / h])
    return {
        "box_iou": (1 - intersection / (area + truth_area - intersection).clamp_min(1e-8)).mean(),
        "centroid_fraction": (
            torch.linalg.vector_norm(predicted[:, :2] - source_boxes[:, :2], dim=1) / diagonal
        ).mean(),
        "log_area_ratio": torch.log((area + 1e-8) / (truth_area + 1e-8)).abs().mean(),
        "structural_edge_ncc": 1 - masked_edge_ncc(source, target, field).mean(),
        "flow_smoothness": (field[:, :, 1:] - field[:, :, :-1]).abs().mean()
        + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean(),
        "flow_fold": F.relu(0.2 - jacobian_determinant(field)).mean(),
        "flow_boundary": F.relu(sampling_map(field).abs() - limit).mean(),
    }


def _cycle_loss(first, second, boxes):
    residual, valid = cycle_field(first, second)
    # Invalid regions must not make the loss zero; boundary loss also supplies gradients.
    norm = torch.linalg.vector_norm(residual, dim=-1)
    global_loss = torch.where(valid, norm, norm + 1).mean()
    points = box_points(boxes)
    residual, valid = cycle_at(first, second, points)
    diagonal = 2 * torch.linalg.vector_norm(boxes[:, 2:], dim=1).clamp_min(1e-5)
    local_norm = torch.linalg.vector_norm(residual, dim=-1) / diagonal[:, None]
    local_loss = torch.where(valid, local_norm, local_norm + 1).mean()
    return global_loss, local_loss


def bidirectional_geometry_loss(visible, infrared, visible_boxes, infrared_boxes, forward, reverse):
    forward, reverse = from_superfusion(forward), from_superfusion(reverse)
    f = _direction_loss(visible, infrared, visible_boxes, infrared_boxes, forward)
    r = _direction_loss(infrared, visible, infrared_boxes, visible_boxes, reverse)
    values = {key: (f[key] + r[key]) / 2 for key in f}
    global_f, roi_f = _cycle_loss(forward, reverse, infrared_boxes)
    global_r, roi_r = _cycle_loss(reverse, forward, visible_boxes)
    values["inverse_consistency"] = (global_f + global_r) / 2
    values["roi_inverse_consistency"] = (roi_f + roi_r) / 2
    total = sum(LOSS_WEIGHTS[key] * value for key, value in values.items())
    return total, {
        **{key: float(value.detach()) for key, value in values.items()},
        "total": float(total.detach()),
    }
