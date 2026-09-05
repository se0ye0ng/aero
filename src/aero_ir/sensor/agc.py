"""Automatic gain control and bit-depth quantisation.

This is the stage most often ignored and most consequential. The detector never sees scene
radiance; it sees an 8-bit image whose mapping was chosen per frame by an AGC. Two frames
with identical physical contrast can land at very different displayed contrast, and a
generator trained on displayed imagery has no reason to preserve the underlying radiometry.

Quantisation is implemented with a straight-through estimator so the chain remains usable
for the N2 track. ``soft_quantise`` is available when a smoother surrogate is needed.
"""

from __future__ import annotations

import torch
from torch import nn


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        return grad_output


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient."""
    return _RoundSTE.apply(x)


class AGCQuantise(nn.Module):
    """Percentile or min-max gain control followed by uniform quantisation.

    Args:
        mode: ``percentile`` (robust, the common choice in fielded systems) or ``minmax``.
        low_percentile / high_percentile: clip points for ``percentile`` mode.
        bit_depth: output bit depth. 8 matches the imagery most IR detection pipelines
            actually train on.
        per_image: compute the mapping per image (as a fielded AGC does) rather than per
            batch.
        soft_quantise: replace the straight-through round with a differentiable surrogate.
    """

    def __init__(
        self,
        mode: str = "percentile",
        low_percentile: float = 0.5,
        high_percentile: float = 99.5,
        bit_depth: int = 8,
        per_image: bool = True,
        soft_quantise: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"percentile", "minmax"}:
            raise ValueError("mode must be percentile or minmax")
        self.mode = mode
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)
        self.bit_depth = int(bit_depth)
        self.per_image = bool(per_image)
        self.soft_quantise = bool(soft_quantise)

    def _limits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = x.flatten(1) if self.per_image else x.flatten().unsqueeze(0)
        if self.mode == "minmax":
            lo = flat.min(dim=1, keepdim=True).values
            hi = flat.max(dim=1, keepdim=True).values
        else:
            q = torch.tensor(
                [self.low_percentile / 100.0, self.high_percentile / 100.0],
                device=x.device,
                dtype=x.dtype,
            )
            qs = torch.quantile(flat, q, dim=1)
            lo, hi = qs[0].unsqueeze(1), qs[1].unsqueeze(1)
        shape = (-1,) + (1,) * (x.dim() - 1) if self.per_image else (1,) * x.dim()
        return lo.reshape(shape), hi.reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self._limits(x)
        scale = (hi - lo).clamp_min(torch.finfo(x.dtype).eps)
        y = ((x - lo) / scale).clamp(0.0, 1.0)
        levels = float(2**self.bit_depth - 1)
        if self.soft_quantise:
            y = y + (torch.round(y * levels) / levels - y).detach() * 0.5
        else:
            y = round_ste(y * levels) / levels
        return y
