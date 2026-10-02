"""FP32 optimizer step for the experimental v5 registration repair objective."""

from __future__ import annotations

import math

import torch

from aero_ir.registration.protocol_v5 import bidirectional_geometry_loss


def training_step(
    model, optimizer, visible, infrared, visible_boxes, infrared_boxes, *, diagnostics=False
):
    optimizer.zero_grad(set_to_none=True)
    forward = model(infrared, visible, direction="visible_to_infrared")
    reverse = model(infrared, visible, direction="infrared_to_visible")
    for name, field in (("forward", forward), ("reverse", reverse)):
        if field.shape != (visible.shape[0], 2, *visible.shape[2:]) or not torch.isfinite(
            field
        ).all():
            raise FloatingPointError(f"invalid {name} field")
        if diagnostics:
            field.retain_grad()
    loss, values = bidirectional_geometry_loss(
        visible, infrared, visible_boxes, infrared_boxes, forward, reverse
    )
    if not torch.isfinite(loss) or not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite v5 loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    checks = {"pre_clip_gradient_norm": float(norm)}
    if diagnostics:
        for name, field in (("forward", forward), ("reverse", reverse)):
            gradient = field.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"invalid {name} field gradient")
            checks[f"{name}_gradient_max_abs"] = float(gradient.abs().max())
    optimizer.step()
    return values, checks
