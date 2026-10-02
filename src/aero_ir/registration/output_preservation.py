"""Preserve a frozen registration teacher on replay observations, not physical GT.

Teacher maps may be wrong despite passing weak box/geometry checks. This loss
only limits drift; it cannot independently qualify a registration method.
"""

import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import box_points, map_points, sampling_map, valid_support

GLOBAL_WEIGHT = 0.05


def direction_preservation(student, teacher, boxes):
    if student.shape != teacher.shape or student.ndim != 4 or student.shape[1] != 2:
        raise ValueError("matching N,2,H,W fields required")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("nonfinite preservation field")
    n, _, h, w = student.shape
    if boxes.shape != (n, 4):
        raise ValueError("one normalized cxcywh box per field required")
    teacher = teacher.detach()
    points = box_points(boxes.detach())
    with torch.no_grad():
        teacher_points = map_points(teacher, points)
        roi_valid = valid_support(points, h, w) & valid_support(teacher_points, h, w)
        mask = valid_support(sampling_map(teacher), h, w)
        counts = mask.sum((1, 2))
        if not roi_valid.all() or (counts == 0).any():
            raise ValueError(
                "replay teacher requires supported ROI points and nonempty global support"
            )
    units = student.new_tensor([w / 2, h / 2])
    roi_error = (map_points(student, points) - teacher_points) * units
    roi_loss = F.smooth_l1_loss(roi_error, torch.zeros_like(roi_error), reduction="none").mean()
    global_error = (student - teacher).permute(0, 2, 3, 1) * units
    global_error = F.smooth_l1_loss(
        global_error, torch.zeros_like(global_error), reduction="none"
    ).mean(-1)
    global_loss = ((global_error * mask).sum((1, 2)) / counts).mean()
    return roi_loss + GLOBAL_WEIGHT * global_loss, {
        "roi_loss": float(roi_loss.detach()),
        "global_loss": float(global_loss.detach()),
        "global_observed_pixels": counts.tolist(),
        "roi_points_per_observation": points.shape[1],
    }


def preservation_loss(student_fields, teacher_fields, visible_boxes, infrared_boxes):
    if len(student_fields) != 2 or len(teacher_fields) != 2:
        raise ValueError("both registration directions required")
    f, fm = direction_preservation(student_fields[0], teacher_fields[0], infrared_boxes)
    r, rm = direction_preservation(student_fields[1], teacher_fields[1], visible_boxes)
    return (f + r) / 2, {"ir_to_rgb": fm, "rgb_to_ir": rm}
