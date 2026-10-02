"""Differentiable box-gate violation candidate, not a correspondence qualifier.

Annotations supervise the loss only. The existing geometry loss remains active,
including support, cycle and Jacobian constraints. A zero auxiliary loss means
only three box predicates passed; it is never full registration approval.
"""

import torch

from aero_ir.registration.geometry import map_boxes
from aero_ir.registration.qualification_v4 import THRESHOLDS


def box_violations(predicted, truth):
    """Return dimensionless per-pair violations of the unchanged box thresholds."""
    if (predicted.ndim != 2 or predicted.shape[1] != 4 or not len(predicted)
            or predicted.shape != truth.shape or predicted.device != truth.device
            or predicted.dtype != truth.dtype):
        raise ValueError("matching nonempty N,4 boxes on the same device/dtype required")
    for value in (predicted, truth):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("finite floating boxes required")
    if (predicted[:, 2:] < 0).any() or (truth[:, 2:] <= 0).any():
        raise ValueError("nonnegative predicted and positive truth extents required")
    truth = truth.detach()
    low, high = predicted[:, :2] - predicted[:, 2:] / 2, predicted[:, :2] + predicted[:, 2:] / 2
    tlow, thigh = truth[:, :2] - truth[:, 2:] / 2, truth[:, :2] + truth[:, 2:] / 2
    overlap = (torch.minimum(high, thigh) - torch.maximum(low, tlow)).clamp_min(0).prod(1)
    area, tarea = predicted[:, 2:].prod(1), truth[:, 2:].prod(1)
    iou = overlap / (area + tarea - overlap).clamp_min(1e-12)
    centroid = torch.linalg.vector_norm(predicted[:, :2] - truth[:, :2], dim=1)
    centroid = centroid / torch.linalg.vector_norm(truth[:, 2:], dim=1).clamp_min(1e-12)
    area_change = (area / tarea.clamp_min(1e-12) - 1).abs()
    t = THRESHOLDS
    return {
        "bbox_iou": (1 - iou / t["minimum_bbox_iou"]).clamp_min(0),
        "centroid_shift_fraction": (
            centroid / t["maximum_centroid_shift_fraction"] - 1
        ).clamp_min(0),
        "absolute_area_ratio_change": (
            area_change / t["maximum_absolute_area_ratio_change"] - 1
        ).clamp_min(0),
    }


def alignment_violation(fields, visible_boxes, infrared_boxes):
    """Mean of the worst normalized box violation per pair across both directions.

The reduction is across six component/direction entries *within each pair*,
then across all pairs. It does not discard easy pairs or change a denominator.
    """
    forward, reverse = fields
    visible_boxes, infrared_boxes = visible_boxes.detach(), infrared_boxes.detach()
    a = box_violations(map_boxes(infrared_boxes, forward), visible_boxes)
    b = box_violations(map_boxes(visible_boxes, reverse), infrared_boxes)
    entries = torch.stack([*a.values(), *b.values()], dim=1)
    return entries.amax(dim=1).mean()
