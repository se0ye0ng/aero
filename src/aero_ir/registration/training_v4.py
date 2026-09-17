"""Shared v4 optimization step for full training and bounded GPU engineering smoke."""

from __future__ import annotations

import math

import torch

from aero_ir.registration.protocol_v4 import bidirectional_geometry_loss


def training_step(
    model, optimizer, visible, infrared, visible_boxes, infrared_boxes, *, diagnostics=False
):
    """Run the actual FP32 two-direction loss/backward/clip/AdamW path.

    Diagnostic tensor snapshots exist only for the short smoke, never full training.
    No checkpoint is written here. A failed step must not be resumed as a valid run.
    """
    optimizer.zero_grad(set_to_none=True)
    forward = model(infrared, visible, direction="visible_to_infrared")
    reverse = model(infrared, visible, direction="infrared_to_visible")
    for name, field in (("forward", forward), ("reverse", reverse)):
        if field.shape != (visible.shape[0], 2, *visible.shape[2:]):
            raise ValueError(f"unexpected {name} field shape: {tuple(field.shape)}")
        if diagnostics:
            if not torch.isfinite(field).all():
                raise FloatingPointError(f"non-finite {name} field")
            field.retain_grad()
    loss, values = bidirectional_geometry_loss(
        visible, infrared, visible_boxes, infrared_boxes, forward, reverse
    )
    if not torch.isfinite(loss) or not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite v4 loss or component")
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
    checks = {}
    before = []
    if diagnostics:
        for name, field in (("forward", forward), ("reverse", reverse)):
            gradient = field.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"missing or non-finite {name} field gradient")
            magnitude = float(gradient.abs().max())
            if magnitude == 0:
                raise RuntimeError(f"{name} field has zero gradient in smoke")
            checks[f"{name}_gradient_max_abs"] = magnitude
        checks["pre_clip_gradient_norm"] = float(norm)
        checks["parameters_with_gradient"] = sum(p.grad is not None for p in parameters)
        before = [p.detach().clone() for p in parameters]
    optimizer.step()
    if diagnostics:
        if not all(torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("non-finite model parameters after optimizer step")
        tensors = [
            v
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        ]
        if not tensors or not all(torch.isfinite(v).all() for v in tensors):
            raise FloatingPointError("missing or non-finite optimizer state")
        update = max(
            float((p.detach() - old).abs().max()) for p, old in zip(parameters, before, strict=True)
        )
        if not math.isfinite(update) or update == 0:
            raise RuntimeError("optimizer did not produce a finite nonzero parameter update")
        checks.update(
            {
                "parameter_update_max_abs": update,
                "parameters_finite": True,
                "optimizer_state_finite": True,
            }
        )
    return values, checks
