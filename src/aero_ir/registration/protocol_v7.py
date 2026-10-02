"""Experimental shared-velocity training; unchanged qualification, no pseudo-GT.

Geometry and geometry+MIND are matched-budget arms. MIND is a noisy visual loss,
not physical correspondence truth. Invalid image support cannot improve its loss.
"""

import torch
from torch import nn

from aero_ir.registration.geometry import centre_grid, from_superfusion, warp
from aero_ir.registration.mind import mind_descriptor
from aero_ir.registration.protocol_v4 import _direction_loss
from aero_ir.registration.protocol_v5 import LOSS_WEIGHTS as V5_WEIGHTS
from aero_ir.registration.protocol_v5 import _direction_constraints
from aero_ir.registration.shared_velocity import shared_fields

ARCHITECTURE = "shared_stationary_velocity_superfusion_v7"
INTEGRATION_STEPS = 7
LEARNING_RATE = 1e-5
MIND_WEIGHT = 0.05
TEXTURE_SSD = 1e-6
LOSS_WEIGHTS = {k: v for k, v in V5_WEIGHTS.items() if k != "structural_edge_ncc"}


class SharedVelocityMatcher(nn.Module):
    """One image-conditioned velocity prediction; exp(+v) and exp(-v) share weights.

    The initial predictor is a v6 displacement network, not a pre-fit velocity
    network. The initialization regression is measured, not hidden by an inverse.
    fields() exposes centre-lattice fields; forward() adapts once for legacy audit.
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def fields(self, infrared, visible):
        raw = self.predictor(infrared, visible, direction="visible_to_infrared")
        velocity = from_superfusion(raw)
        return shared_fields(velocity, INTEGRATION_STEPS)

    def forward(self, infrared, visible, *, direction):
        if direction not in ("visible_to_infrared", "infrared_to_visible"):
            raise ValueError("unknown registration direction")
        first, second = self.fields(infrared, visible)
        field = first if direction == "visible_to_infrared" else second
        return field - from_superfusion(torch.zeros_like(field))


def observation(image, boxes, excluded):
    """Descriptors and fixed target observation mask (expanded train box + texture).

    Exclusions cover heuristic HUD and descriptor boundaries. Boxes are existing
    weak train annotations, never fine-grained correspondence labels.
    """
    descriptor, texture = mind_descriptor(image)
    n, _, h, w = image.shape
    if excluded.shape != (n, h, w) or excluded.dtype != torch.bool:
        raise ValueError("expected boolean N,H,W exclusions")
    safe = ~excluded.clone()
    safe[:, :3] = safe[:, -3:] = False
    safe[:, :, :3] = safe[:, :, -3:] = False
    grid = (centre_grid(image.new_zeros(n, 2, h, w)) + 1) / 2
    # Existing train box expanded to 1.5 times width/height; no val/test box use.
    roi = ((grid - boxes[:, None, None, :2]).abs() <= boxes[:, None, None, 2:] * 0.75).all(-1)
    textured = texture[:, 0] > TEXTURE_SSD
    return {
        "descriptor": descriptor.detach(),
        "safe": safe & textured,
        "observe": safe & textured & roi,
    }


def descriptor_direction(source, target, field):
    fixed = target["observe"]
    count = fixed.sum((1, 2))
    eligible = count > 0
    sampled_safe = warp(source["safe"][:, None].float(), field)[:, 0] >= 1 - 1e-6
    error = (warp(source["descriptor"], field) - target["descriptor"]).square().mean(1)
    # Descriptor entries lie in [0,1]; missing source evidence receives worst loss1,
    # with a fixed denominator. It cannot make the fit look better by disappearing.
    error = torch.where(sampled_safe, error, torch.ones_like(error))
    per_pair = (error * fixed).sum((1, 2)) / count.clamp_min(1)
    loss = per_pair[eligible].mean() if eligible.any() else field.sum() * 0
    coverage = ((sampled_safe & fixed).sum((1, 2)) / count.clamp_min(1))[eligible]
    return loss, {
        "eligible_fraction": float(eligible.float().mean()),
        "source_observed_fraction": float(coverage.mean()) if coverage.numel() else 0.0,
        "target_observed_pixels": int(count.sum()),
    }


def training_loss(visible, infrared, visible_boxes, infrared_boxes, fields, *, observations=None):
    """Consume centre fields directly; do not call from_superfusion twice."""
    forward, reverse = fields
    f = _direction_loss(visible, infrared, visible_boxes, infrared_boxes, forward)
    r = _direction_loss(infrared, visible, infrared_boxes, visible_boxes, reverse)
    values = {k: (f[k] + r[k]) / 2 for k in f if k in LOSS_WEIGHTS}
    f = _direction_constraints(forward, reverse, infrared_boxes)
    r = _direction_constraints(reverse, forward, visible_boxes)
    values.update({k: (f[k] + r[k]) / 2 for k in f})
    total = sum(LOSS_WEIGHTS[k] * v for k, v in values.items())
    diagnostics = {}
    if observations is not None:
        visible_obs, ir_obs = observations
        a, da = descriptor_direction(visible_obs, ir_obs, forward)
        b, db = descriptor_direction(ir_obs, visible_obs, reverse)
        values["mind"] = (a + b) / 2
        total = total + MIND_WEIGHT * values["mind"]
        diagnostics = {f"mind_{k}": (da[k] + db[k]) / 2 for k in da}
    return total, {
        **{k: float(v.detach()) for k, v in values.items()},
        **diagnostics,
        "total": float(total.detach()),
    }


def step(model, optimizer, visible, infrared, visible_boxes, infrared_boxes, *, observations=None):
    optimizer.zero_grad(set_to_none=True)
    fields = model.fields(infrared, visible)
    total, metrics = training_loss(
        visible, infrared, visible_boxes, infrared_boxes, fields, observations=observations
    )
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite v7 objective")
    total.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    metrics["gradient_norm"] = float(norm)
    return metrics
