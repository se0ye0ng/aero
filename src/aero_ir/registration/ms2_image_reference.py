"""Sparse calibrated reference masks for an image-only MS2 matcher diagnostic.

These references assume calibration, axial depth and the declared motion model.
Depth-product agreement is not independent confirmation of visibility or image GT.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from aero_ir.registration.calibrated import (
    ms2_depth_metres,
    project_pixels,
    zbuffer_candidates,
)


def thermal_window(images: list[np.ndarray]) -> tuple[float, float]:
    if not images or any(im.dtype != np.uint16 or im.ndim != 2 or not im.size for im in images):
        raise ValueError("nonempty native uint16 thermal images required")
    low, high = np.percentile(np.concatenate([im.ravel() for im in images]), [.5, 99.5])
    if high <= low:
        raise ValueError("degenerate training-panel thermal window")
    return float(low), float(high)


def display_thermal(image: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    low, high = window
    if (image.dtype != np.uint16 or image.ndim != 2 or not np.isfinite(window).all()
            or high <= low):
        raise ValueError("invalid thermal image or fixed window")
    # Matching input only. Raw sensor counts and RFS/generator targets are untouched.
    return np.rint(np.clip((image.astype(np.float64) - low) / (high - low), 0, 1)*255).astype(
        np.uint8)


def sparse_reference(source_depth, target_depth, source_k, target_k, transform):
    source, target = ms2_depth_metres(source_depth), ms2_depth_metres(target_depth)
    y, x = np.nonzero(np.isfinite(source))
    xy = np.column_stack((x, y)).astype(np.float64)
    projection = project_pixels(xy, source[y, x], source_k, target_k, transform,
                                source_shape=source.shape, target_shape=target.shape)
    zmask = zbuffer_candidates(projection, target.shape)
    # The optional consistency slice is fixed before examining any image matches.
    # Report the broader masks too; do not turn this slice into a qualification gate.
    dmask = np.zeros(len(xy), dtype=bool)
    ty, tx = np.nonzero(np.isfinite(target))
    indices = np.flatnonzero(zmask)
    if len(tx) and len(indices):
        distance, nearest = cKDTree(np.column_stack((tx, ty))).query(projection.target_xy[indices])
        delta = np.abs(projection.target_z_m[indices] - target[ty[nearest], tx[nearest]])
        dmask[indices] = (distance <= 1.) & (delta <= .03)
    return dict(source_xy=xy, target_xy=projection.target_xy,
                geometric_mask=projection.supported, zbuffer_mask=zmask,
                depth_consistency_mask=dmask)


def score_map(predicted: np.ndarray, reference: dict) -> dict:
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != reference["target_xy"].shape:
        raise ValueError("prediction/reference shape mismatch")
    errors = np.linalg.norm(predicted - reference["target_xy"], axis=1)
    errors[~np.isfinite(errors)] = np.inf
    result = dict(all_source_depth_points=len(errors), masks={})
    for name in ("geometric_mask", "zbuffer_mask", "depth_consistency_mask"):
        mask = reference[name]
        selected = errors[mask]
        finite = selected[np.isfinite(selected)]
        result["masks"][name] = dict(
            references=len(selected), unsupported_predictions=int((~np.isfinite(selected)).sum()),
            conditional_finite_median_px=float(np.median(finite)) if len(finite) else None,
            conditional_finite_p95_px=float(np.percentile(finite, 95)) if len(finite) else None,
            fraction_within_target_native_px={str(t): float(np.mean(selected <= t))
                                              if len(selected) else None
                                              for t in (1., 3., 5., 10.)})
    return result
