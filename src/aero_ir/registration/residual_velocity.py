"""Image-conditioned residual SVF candidate, not a qualified registration model.

No annotations, masks fitted from boxes, sequence IDs or optimized control targets
enter prediction. Existing train boxes may supervise the caller's training loss.
The frozen v7 predictor supplies the starting velocity; zero initialization
preserves both of its maps exactly. A new checkpoint type is needed for this head.
"""

import torch
from torch import nn

from aero_ir.registration.geometry import centre_grid, from_superfusion, warp
from aero_ir.registration.protocol_v7 import INTEGRATION_STEPS
from aero_ir.registration.shared_velocity import control_velocity, integrate_velocity, shared_fields

ARCHITECTURE = "image_conditioned_residual_svf_candidate_01"
CONTEXT_CHANNELS = 14
MAX_RESIDUAL_PIXELS = 24.0


def image_context(visible, infrared, base_velocity):
    """Images, frozen velocity, source warp/support and coordinates; no boxes.

    Caller must supply same-grid normalized-centre velocity and RGB images in
    [0,1]. The black padded source is accompanied by its fractional support map.
    This context is detached, never a path for updating the frozen predictor.
    """
    if (visible.shape != infrared.shape or visible.ndim != 4 or visible.shape[1] != 3
            or base_velocity.shape != (visible.shape[0], 2, *visible.shape[2:])
            or min(visible.shape[2:]) < 32):
        raise ValueError("expected matching N,3,H,W images and N,2,H,W velocity, H,W>=32")
    for value in (visible, infrared, base_velocity):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("context inputs must be finite floating tensors")
    if any(v.min() < 0 or v.max() > 1 for v in (visible, infrared)):
        raise ValueError("images must lie in [0,1]")
    with torch.no_grad():
        forward = integrate_velocity(base_velocity, INTEGRATION_STEPS)
        source = warp(visible, forward)
        support = warp(torch.ones_like(visible[:, :1]), forward)
        coordinates = centre_grid(base_velocity).permute(0, 3, 1, 2).expand(
            visible.shape[0], -1, -1, -1)
        return torch.cat((visible * 2 - 1, infrared * 2 - 1, source * 2 - 1,
                          base_velocity, support, coordinates), dim=1).detach()


class ResidualVelocityHead(nn.Module):
    """Predict roughly16x16 controls for256x256 inputs, then integrate jointly."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = CONTEXT_CHANNELS
        for width in (16, 24, 32, 48):
            layers.extend((nn.Conv2d(channels, width, 3, stride=2, padding=1),
                           nn.GroupNorm(4, width), nn.SiLU()))
            channels = width
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Conv2d(channels, 2, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context):
        if context.ndim != 4 or context.shape[1] != CONTEXT_CHANNELS:
            raise ValueError("expected image-only registration context")
        return self.output(self.encoder(context))

    def fields(self, context, base_velocity):
        controls = self(context)
        residual = control_velocity(controls, base_velocity.shape[2:],
                                    max_pixels=MAX_RESIDUAL_PIXELS)
        return shared_fields(base_velocity.detach() + residual, INTEGRATION_STEPS)


class ResidualVelocityMatcher(nn.Module):
    """Image-only inference wrapper with an explicitly frozen v7 initializer."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model.eval().requires_grad_(False)
        self.head = ResidualVelocityHead()

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def fields(self, infrared, visible):
        with torch.no_grad():
            base = from_superfusion(self.base_model.predictor(
                infrared, visible, direction="visible_to_infrared"))
        return self.head.fields(image_context(visible, infrared, base), base)
