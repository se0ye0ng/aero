"""Fixed-pattern noise and non-uniformity-correction residual.

Real focal-plane arrays leave a spatially fixed, temporally stable pattern after NUC:
a column component from the readout integrated circuit, a per-pixel component from
detector-to-detector variation, and a slow gain residual. Generated imagery has none of it,
and a detector trained without it learns a noise model the sensor will not supply.

The pattern is drawn once from a fixed seed and registered as a buffer, so it is stable
across a run and across resumes - which is what makes it *fixed* pattern noise rather than
a second source of temporal noise.
"""

from __future__ import annotations

import torch
from torch import nn


class FixedPatternNoise(nn.Module):
    """Column + per-pixel additive offset and a multiplicative gain residual.

    Args:
        column_sigma_dn: standard deviation of the per-column offset.
        pixel_sigma_dn: standard deviation of the per-pixel offset.
        nuc_residual_gain_sigma: standard deviation of the per-pixel multiplicative gain
            error remaining after non-uniformity correction.
        shape: (H, W) of the focal plane. If None, the pattern is generated lazily on the
            first forward pass and cached.
        seed: pattern seed. Fixed by default so the pattern is reproducible.
    """

    def __init__(
        self,
        column_sigma_dn: float = 0.8,
        pixel_sigma_dn: float = 0.5,
        nuc_residual_gain_sigma: float = 0.004,
        shape: tuple[int, int] | None = None,
        seed: int = 12345,
    ) -> None:
        super().__init__()
        self.column_sigma_dn = float(column_sigma_dn)
        self.pixel_sigma_dn = float(pixel_sigma_dn)
        self.nuc_residual_gain_sigma = float(nuc_residual_gain_sigma)
        self.seed = int(seed)
        self.register_buffer("offset", torch.empty(0), persistent=True)
        self.register_buffer("gain", torch.empty(0), persistent=True)
        if shape is not None:
            self._build(shape, torch.device("cpu"), torch.float32)

    def _build(self, shape: tuple[int, int], device: torch.device, dtype: torch.dtype) -> None:
        h, w = shape
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        col = torch.randn(1, w, generator=g) * self.column_sigma_dn
        pix = torch.randn(h, w, generator=g) * self.pixel_sigma_dn
        gain = 1.0 + torch.randn(h, w, generator=g) * self.nuc_residual_gain_sigma
        self.offset = (col + pix).to(device=device, dtype=dtype)
        self.gain = gain.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.offset.numel() == 0 or self.offset.shape != (h, w):
            self._build((h, w), x.device, x.dtype)
        return x * self.gain + self.offset
