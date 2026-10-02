"""Small, differentiable 2D MIND-style descriptor for a bounded diagnostic.

Based on Heinrich et al., MedIA 2012, doi:10.1016/j.media.2012.05.008.
This is an explicitly defined 2D adaptation, NOT the original 3D MIND/SSC code:
four axial neighbours at each of radii 1 and 2, 3x3 uniform patch SSD,
local mean SSD normalization and per-radius maximum normalization.
Flat regions have finite descriptors but contain no correspondence evidence.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] not in (1, 3):
        raise ValueError("expected N,1/3,H,W image")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("image must be finite floating point")
    if image.shape[1] == 1:
        return image
    return (image * image.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(
        1, keepdim=True
    )


def mind_descriptor(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return N,8,H,W descriptor and N,1,H,W mean local SSD (texture proxy).

    A three-pixel boundary must be excluded from correspondence scoring.
    Expected intensity units are [0,1]. The epsilon is a numerical floor,
    not evidence of texture or a registration confidence calibration.
    """
    gray = grayscale(image)
    h, w = gray.shape[-2:]
    if min(h, w) < 8:
        raise ValueError("MIND requires H,W >= 8")
    descriptors, variances = [], []
    for radius in (1, 2):
        padded = F.pad(gray, (radius,) * 4, mode="replicate")
        distances = []
        for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
            shifted = padded[:, :, radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            distances.append(F.avg_pool2d((gray - shifted).square(), 3, stride=1, padding=1))
        distances = torch.cat(distances, dim=1)
        variance = distances.mean(1, keepdim=True)
        centered = distances - distances.amin(1, keepdim=True)
        descriptors.append(torch.exp(-centered / variance.clamp_min(1e-8)))
        variances.append(variance)
    return torch.cat(descriptors, dim=1), torch.stack(variances).mean(0)


def masked_descriptor_ssd(source, target, field, mask):
    """Per-sample loss on a caller-specified, fixed supported mask.

    Compute descriptors BEFORE warping. The caller must exclude padded source
    and target support; empty masks raise rather than yielding a perfect score.
    """
    from aero_ir.registration.geometry import warp

    if source.shape != target.shape or source.shape[1] != 8:
        raise ValueError("expected matching N,8,H,W descriptors")
    if mask.shape != source[:, 0].shape or mask.dtype != torch.bool:
        raise ValueError("expected boolean N,H,W mask")
    count = mask.sum((1, 2))
    if (count == 0).any():
        raise ValueError("empty descriptor support")
    error = (warp(source, field) - target).square().mean(1)
    return (error * mask).sum((1, 2)) / count
