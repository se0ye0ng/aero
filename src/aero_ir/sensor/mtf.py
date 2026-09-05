"""Optical / detector modulation transfer function as a separable Gaussian blur.

A real MTF is the product of diffraction, aberration, detector-footprint and motion terms.
For training-data realism the dominant effect is a low-pass with a characteristic cutoff, so
a Gaussian whose sigma is derived from the cutoff frequency is an adequate and differentiable
stand-in. The point is not to model a specific optic; it is to stop the generator from
producing edges no infrared optic can form.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sigma_from_cutoff(cutoff_cycles_per_pixel: float) -> float:
    """Gaussian sigma (pixels) whose MTF falls to 1/e at the given cutoff frequency."""
    if cutoff_cycles_per_pixel <= 0:
        raise ValueError("cutoff_cycles_per_pixel must be positive")
    # A Gaussian PSF has MTF(f) = exp(-2*pi^2*sigma^2*f^2).
    return 1.0 / (math.sqrt(2.0) * math.pi * cutoff_cycles_per_pixel)


class MTFBlur(nn.Module):
    """Separable Gaussian low-pass approximating the sensor MTF.

    Args:
        cutoff_cycles_per_pixel: characteristic optical cutoff.
        detector_fill_factor: fill factor in (0, 1]; a lower factor widens the effective
            detector footprint and therefore the blur.
        learnable: expose the cutoff as a parameter for the N2 fitting track.
    """

    def __init__(
        self,
        cutoff_cycles_per_pixel: float = 0.35,
        detector_fill_factor: float = 1.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < detector_fill_factor <= 1.0:
            raise ValueError("detector_fill_factor must be in (0, 1]")
        sigma = sigma_from_cutoff(cutoff_cycles_per_pixel) / math.sqrt(detector_fill_factor)
        log_sigma = torch.tensor(float(math.log(sigma)))
        self.log_sigma = (
            nn.Parameter(log_sigma) if learnable else nn.Parameter(log_sigma, requires_grad=False)
        )

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def _kernel(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        sigma = self.sigma.to(device=device, dtype=dtype)
        radius = int(max(1, math.ceil(3.0 * float(sigma.detach()))))
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        return k / k.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) float tensor in scene units."""
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        k = self._kernel(x.dtype, x.device)
        c = x.shape[1]
        pad = (k.numel() - 1) // 2
        kh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
        kv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
        x = nn.functional.conv2d(
            nn.functional.pad(x, (pad, pad, 0, 0), mode="reflect"), kh, groups=c
        )
        x = nn.functional.conv2d(
            nn.functional.pad(x, (0, 0, pad, pad), mode="reflect"), kv, groups=c
        )
        return x
