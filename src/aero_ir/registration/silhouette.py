"""Reviewed silhouette diagnostics, not physical pixel correspondence accuracy."""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.nn import functional as F


def binary_mask(value, name="mask"):
    value = np.asarray(value)
    if value.ndim != 2 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 2D array")
    if not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must contain binary 0/1 values")
    return value.astype(bool)


def silhouette_scores(source_mask, target_mask, maps):
    """Score independent Dice and symmetric boundary Chamfer on common support.

    Maps are target-output to source-input normalized pixel-center sampling maps,
    not displacement fields. Chamfer units are target-image pixels. Empty masks
    or clipped contours abstain rather than receiving perfect-empty scores.
    """
    source = binary_mask(source_mask, "source_mask")
    target = binary_mask(target_mask, "target_mask")
    maps = torch.as_tensor(maps, dtype=torch.float32)
    if (
        maps.ndim != 4
        or maps.shape[-1] != 2
        or maps.shape[0] < 1
        or tuple(maps.shape[1:3]) != target.shape
        or not torch.isfinite(maps).all()
    ):
        raise ValueError("maps must be finite Nx(target height)x(target width)x2")
    hs, ws = source.shape
    limits = maps.new_tensor((1 - 1 / ws, 1 - 1 / hs))
    support = (maps.abs() <= limits).all(-1).all(0).cpu().numpy()
    safe = binary_erosion(support, structure=np.ones((3, 3)), border_value=0)
    results = []
    source_tensor = torch.from_numpy(source.astype(np.float32))[None, None]
    warped = (
        F.grid_sample(
            source_tensor.expand(len(maps), -1, -1, -1), maps, mode="bilinear", align_corners=False
        )[:, 0]
        .cpu()
        .numpy()
    )
    for soft in warped:
        hard = soft >= 0.5
        if not source.any() or not target.any() or not hard.any():
            results.append(
                {
                    "eligible": False,
                    "reason": "empty_silhouette",
                    "dice_loss": None,
                    "symmetric_boundary_chamfer_px": None,
                }
            )
            continue
        # Do not let common-overlap clipping erase difficult contour sections.
        if np.any(target & ~safe) or np.any(hard & ~safe):
            results.append(
                {
                    "eligible": False,
                    "reason": "contour_touches_invalid_support",
                    "dice_loss": None,
                    "symmetric_boundary_chamfer_px": None,
                }
            )
            continue
        a = hard & ~binary_erosion(hard, structure=np.ones((3, 3)), border_value=0)
        b = target & ~binary_erosion(target, structure=np.ones((3, 3)), border_value=0)
        dice = 1 - 2 * (soft[support] * target[support]).sum() / (
            soft[support].sum() + target[support].sum()
        )
        chamfer = 0.5 * (
            distance_transform_edt(~a)[b].mean() + distance_transform_edt(~b)[a].mean()
        )
        results.append(
            {
                "eligible": True,
                "reason": None,
                "dice_loss": float(dice),
                "symmetric_boundary_chamfer_px": float(chamfer),
            }
        )
    return {
        "common_support_pixels": int(support.sum()),
        "candidates": results,
        "independent_pixel_correspondence_accuracy": False,
    }
