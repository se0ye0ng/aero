"""Small differentiable Gaussian-Parzen MI, not the Mattes estimator."""

from __future__ import annotations

import torch


def joint_pdf(a, b, *, bins=16, bandwidth=None, hard=False):
    """Return batched joint PDFs for paired [B,N] intensities in [0,1]."""
    if a.ndim == 1:
        a = a[None]
    if b.ndim == 1:
        b = b[None]
    if a.ndim != 2 or a.shape != b.shape or a.shape[-1] == 0:
        raise ValueError("expected equal nonempty [batch,pixels] inputs")
    if bins < 2 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("invalid bins or nonfinite intensities")
    if (a < 0).any() or (a > 1).any() or (b < 0).any() or (b > 1).any():
        raise ValueError("intensities must be in [0,1]")
    if hard:
        ai = (a * bins).long().clamp_max(bins - 1)
        bi = (b * bins).long().clamp_max(bins - 1)
        pdf = a.new_zeros((len(a), bins * bins))
        pdf.scatter_add_(1, ai * bins + bi, torch.ones_like(a))
        return pdf.reshape(-1, bins, bins) / a.shape[-1]
    sigma = 0.5 / (bins - 1) if bandwidth is None else bandwidth
    if sigma <= 0:
        raise ValueError("bandwidth must be positive")
    centres = torch.linspace(0, 1, bins, dtype=a.dtype, device=a.device)
    aw = torch.softmax(-0.5 * ((a[..., None] - centres) / sigma).square(), dim=-1)
    bw = torch.softmax(-0.5 * ((b[..., None] - centres) / sigma).square(), dim=-1)
    return aw.transpose(1, 2).bmm(bw) / a.shape[-1]


def mutual_information(a, b, *, bins=16, bandwidth=None, hard=False):
    """MI in nats (maximize); caller supplies fixed spatial sample support."""
    p = joint_pdf(a, b, bins=bins, bandwidth=bandwidth, hard=hard)
    independent = p.sum(-1, keepdim=True) * p.sum(-2, keepdim=True)
    return (p * (p.clamp_min(1e-12).log() - independent.clamp_min(1e-12).log())).sum((-1, -2))
