"""Finite-checked optimizer step with an immutable v4 teacher."""

from __future__ import annotations

import math

import torch

from aero_ir.registration.protocol_v6 import bidirectional_geometry_loss


def training_step(model, teacher, optimizer, visible, infrared, visible_boxes, infrared_boxes):
    if teacher is model or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("v6 requires a separate frozen teacher")
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        teacher_forward = teacher(infrared, visible, direction="visible_to_infrared")
        teacher_reverse = teacher(infrared, visible, direction="infrared_to_visible")
    forward = model(infrared, visible, direction="visible_to_infrared")
    reverse = model(infrared, visible, direction="infrared_to_visible")
    for field in (forward, reverse, teacher_forward, teacher_reverse):
        if field.shape != (visible.shape[0], 2, *visible.shape[2:]) or not torch.isfinite(
            field
        ).all():
            raise FloatingPointError("invalid v6 student/teacher field")
    loss, values = bidirectional_geometry_loss(
        visible, infrared, visible_boxes, infrared_boxes, forward, reverse,
        teacher_forward, teacher_reverse,
    )
    if not torch.isfinite(loss) or not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite v6 loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    return values, {"pre_clip_gradient_norm": float(norm)}
