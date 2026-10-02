"""Depth-product convention diagnostics, not independent image registration GT."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from aero_ir.registration.calibrated import intrinsic_matrix, ms2_depth_metres, project_pixels


def _summary(values: np.ndarray) -> dict:
    if not len(values):
        return dict(count=0, min=None, median=None, p90=None, max=None)
    quantiles = np.percentile(values, [0, 50, 90, 100])
    return dict(count=len(values), **dict(zip(("min", "median", "p90", "max"),
                                            map(float, quantiles), strict=True)))


def depth_product_consistency(source_encoded: np.ndarray, target_encoded: np.ndarray,
                              source_k: np.ndarray, target_k: np.ndarray,
                              target_from_source: np.ndarray,
                              *, association_radius_px: float = 1.) -> dict:
    """Compare two explicit depth conventions without selecting/qualifying either.

    Association is nearest nonzero target pixel, not a known same-return LiDAR
    identity. The products may share the calibration used for projection, so even
    exact agreement cannot independently certify camera-image alignment. Both
    association denominators and missing support are reported, with no hole fill.
    """
    if not np.isfinite(association_radius_px) or association_radius_px <= 0:
        raise ValueError("positive finite association radius required")
    source_k, target_k = intrinsic_matrix(source_k), intrinsic_matrix(target_k)
    source_depth, target_depth = ms2_depth_metres(source_encoded), ms2_depth_metres(target_encoded)
    sy, sx = np.nonzero(np.isfinite(source_depth))
    ty, tx = np.nonzero(np.isfinite(target_depth))
    source, target = np.column_stack((sx, sy)), np.column_stack((tx, ty))
    measured = source_depth[sy, sx]
    target_measured = target_depth[ty, tx]
    rays = np.column_stack((source, np.ones(len(source)))) @ np.linalg.inv(source_k).T
    tree = cKDTree(target) if len(target) else None
    models = {}
    for convention in ("axial_z", "euclidean_range"):
        z = measured if convention == "axial_z" else measured / np.linalg.norm(rays, axis=1)
        projection = project_pixels(source, z, source_k, target_k, target_from_source,
                                    source_shape=source_depth.shape,
                                    target_shape=target_depth.shape)
        indices = np.flatnonzero(projection.supported)
        distances, residuals = np.array([]), np.array([])
        associated = 0
        if tree is not None and len(indices):
            distances, nearest = tree.query(projection.target_xy[indices])
            keep = distances <= association_radius_px
            associated = int(keep.sum())
            predicted = projection.target_z_m[indices]
            if convention == "euclidean_range":
                target_rays = np.column_stack((projection.target_xy[indices],
                                               np.ones(len(indices)))) @ np.linalg.inv(target_k).T
                predicted = predicted * np.linalg.norm(target_rays, axis=1)
            residuals = np.abs(predicted[keep] - target_measured[nearest[keep]])
        models[convention] = dict(
            source_points=len(source), target_points=len(target),
            projected_supported=len(indices), associated_points=associated,
            associated_fraction_of_source=associated / len(source) if len(source) else None,
            nearest_pixel_distance_px=_summary(distances),
            associated_absolute_depth_residual_m=_summary(residuals))
    return dict(association="nearest_nonzero_target_depth_pixel",
                association_radius_px=association_radius_px, conventions=models,
                correspondence_identity_verified=False, independent_image_reference=False,
                registration_qualified=False)
