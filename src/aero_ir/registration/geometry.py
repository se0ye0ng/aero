"""Pixel-centre registration geometry (v4); never pass raw SuperFusion fields here.

Public fields are N,2,H,W normalized displacements on an align_corners=False
pixel-centre lattice. They map output/target points to input/source points.
The legacy network is deliberately unchanged: convert its output exactly once
with ``from_superfusion``. This preserves its actual image sampling map, not
the legacy interpretation that a raw zero field means identity.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

CONVENTION = "pixel_centres_align_corners_false_v1"


def _check(field: torch.Tensor) -> None:
    if field.ndim != 4 or field.shape[1] != 2 or min(field.shape[2:]) < 2:
        raise ValueError("field must have shape N,2,H,W with H,W >= 2")
    if not field.is_floating_point():
        raise ValueError("field must be floating point")


def centre_grid(field: torch.Tensor) -> torch.Tensor:
    """Return 1,H,W,2 pixel centres in normalized image-boundary coordinates."""
    _check(field)
    height, width = field.shape[2:]
    x = (torch.arange(width, device=field.device, dtype=field.dtype) + 0.5) * 2 / width - 1
    y = (torch.arange(height, device=field.device, dtype=field.dtype) + 0.5) * 2 / height - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def from_superfusion(raw: torch.Tensor) -> torch.Tensor:
    """Re-express endpoint_grid + raw relative to the pixel-centre grid.

    The resulting image warp is numerically equivalent to the historical warp;
    points, composition and derivatives now refer to that same map.
    """
    _check(raw)
    height, width = raw.shape[2:]
    y = torch.linspace(-1, 1, height, device=raw.device, dtype=raw.dtype)
    x = torch.linspace(-1, 1, width, device=raw.device, dtype=raw.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    endpoints = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    return raw + (endpoints - centre_grid(raw)).permute(0, 3, 1, 2)


def sampling_map(field: torch.Tensor) -> torch.Tensor:
    return centre_grid(field) + field.permute(0, 2, 3, 1)


def warp(source: torch.Tensor, field: torch.Tensor, *, mode: str = "bilinear") -> torch.Tensor:
    if source.ndim != 4 or source.shape[0] != field.shape[0]:
        raise ValueError("source and field batch dimensions must match")
    return F.grid_sample(source, sampling_map(field), mode=mode, align_corners=False)


def map_points(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Map N,P,2 normalized points. Consumers must check valid_support separately.

    Border extension is only a numerical definition outside the centre lattice;
    those points must not count as observed correspondences or valid cycles.
    """
    _check(field)
    if points.ndim != 3 or points.shape[0] != field.shape[0] or points.shape[2] != 2:
        raise ValueError("points must have shape N,P,2 and match field batch")
    sampled = (
        F.grid_sample(field, points.unsqueeze(2), padding_mode="border", align_corners=False)
        .squeeze(-1)
        .transpose(1, 2)
    )
    return points + sampled


def valid_support(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Require full bilinear support (pixel centres), not padded image borders."""
    limit = points.new_tensor([1 - 1 / width, 1 - 1 / height])
    return torch.isfinite(points).all(dim=-1) & (points.abs() <= limit + 1e-6).all(dim=-1)


def box_points(boxes: torch.Tensor, side: int = 9) -> torch.Tensor:
    """Sample a filled box grid, including its perimeter; boxes are [0,1] cxcywh.

    Sampling interiors also catches non-affine extrema missed by perimeter-only
    enclosure. This remains a finite support diagnostic, not a segmentation mask.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4 or side < 2:
        raise ValueError("boxes must have shape N,4 and side must be >= 2")
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("boxes must be finite with positive width and height")
    t = torch.linspace(-0.5, 0.5, side, device=boxes.device, dtype=boxes.dtype)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    offsets = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    return 2 * (boxes[:, None, :2] + offsets * boxes[:, None, 2:]) - 1


def map_boxes(boxes: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    mapped = (map_points(field, box_points(boxes)) + 1) / 2
    low, high = mapped.amin(dim=1), mapped.amax(dim=1)
    return torch.cat(((low + high) / 2, high - low), dim=1)


def cycle_at(
    first: torch.Tensor, second: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return second(first(p))-p and support valid in both domains and on return."""
    _check(first)
    _check(second)
    middle = map_points(first, points)
    returned = map_points(second, middle)
    valid = (
        valid_support(points, *first.shape[2:])
        & valid_support(middle, *second.shape[2:])
        & valid_support(returned, *first.shape[2:])
    )
    return returned - points, valid


def cycle_field(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = centre_grid(first).expand(first.shape[0], -1, -1, -1).flatten(1, 2)
    residual, valid = cycle_at(first, second, points)
    return residual.reshape(first.shape[0], *first.shape[2:], 2), valid.reshape(
        first.shape[0], *first.shape[2:]
    )


def pixel_norm(residual: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return torch.linalg.vector_norm(residual * residual.new_tensor([width / 2, height / 2]), dim=-1)


def jacobian_determinant(field: torch.Tensor) -> torch.Tensor:
    """Jacobian at cell centres using average opposite-edge derivatives.

    Centre spacing is 2/W,2/H, NOT 2/(W-1),2/(H-1).
    """
    grid = sampling_map(field)
    height, width = field.shape[2:]
    dx = (grid[:, :, 1:] - grid[:, :, :-1]) * (width / 2)
    dy = (grid[:, 1:] - grid[:, :-1]) * (height / 2)
    dx = (dx[:, 1:] + dx[:, :-1]) / 2
    dy = (dy[:, :, 1:] + dy[:, :, :-1]) / 2
    return dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]


def point_jacobian_determinant(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Central differences at ROI points; includes validity of all stencil points."""
    height, width = field.shape[2:]
    ex, ey = points.new_tensor([2 / width, 0]), points.new_tensor([0, 2 / height])
    dx = (map_points(field, points + ex) - map_points(field, points - ex)) * (width / 4)
    dy = (map_points(field, points + ey) - map_points(field, points - ey)) * (height / 4)
    determinant = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    valid = torch.ones_like(determinant, dtype=torch.bool)
    for offset in (ex, -ex, ey, -ey):
        valid &= valid_support(points + offset, height, width)
    return determinant.masked_fill(~valid, float("nan"))
