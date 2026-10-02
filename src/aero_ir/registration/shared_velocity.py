"""One stationary velocity, two approximately inverse pixel-centre maps.

Scaling-and-squaring is established registration machinery, not project novelty.
Discrete interpolation does NOT guarantee positive Jacobians or exact inversion;
the unchanged support, Jacobian and cycle measurements must still be evaluated.
All fields here are v4 centre-lattice displacements, never raw SuperFusion fields.
"""

import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import centre_grid, sampling_map


def integrate_velocity(velocity: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """Approximate exp(v) on the same normalized align_corners=False lattice.

    Border extension defines numerical composition only. It does not grant valid
    camera support; use qualification_v4 on the resulting maps before accepting.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 12:
        raise ValueError("integration steps must be an integer in [1, 12]")
    centre_grid(velocity)  # Validate N,2,H,W floating-point shape.
    if not torch.isfinite(velocity).all():
        raise ValueError("nonfinite velocity")
    field = velocity / (2**steps)
    for _ in range(steps):
        sampled = F.grid_sample(
            field,
            sampling_map(field),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        field = field + sampled
    return field


def shared_fields(velocity: torch.Tensor, steps: int = 7):
    """Use exp(v) and exp(-v); the inverse is not simply the negative displacement."""
    return integrate_velocity(velocity, steps), integrate_velocity(-velocity, steps)


def control_velocity(
    controls: torch.Tensor,
    shape: tuple[int, int],
    *,
    max_pixels: float = 24.0,
    taper_boundary: bool = True,
) -> torch.Tensor:
    """Smooth low-resolution controls, bounded in native network-pixel units.

    This only parameterizes a transform; it neither predicts controls from images
    nor uses target boxes. A future image-conditioned predictor can supply controls.
    """
    centre_grid(controls)
    h, w = shape
    if min(h, w) < 3 or not 0 < max_pixels < float("inf"):
        raise ValueError("invalid output shape or velocity bound")
    velocity = F.interpolate(controls, size=shape, mode="bilinear", align_corners=False).tanh()
    if taper_boundary:
        # Exactly zero on the first/last centre rows/columns, not image boundaries.
        x = torch.linspace(0, 1, w, device=controls.device, dtype=controls.dtype)
        y = torch.linspace(0, 1, h, device=controls.device, dtype=controls.dtype)
        envelope = (16 * y * (1 - y))[:, None] * (x * (1 - x))[None, :]
        velocity = velocity * envelope[None, None]
    scale = controls.new_tensor([2 * max_pixels / w, 2 * max_pixels / h])
    return velocity * scale[None, :, None, None]
