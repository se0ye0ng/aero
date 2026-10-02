"""Pinhole reprojection primitives for independently supplied calibration/depth.

Coordinates are native pixel centres (top-left centre is 0,0), camera points are
column vectors, depth is camera-axis Z in metres, and T_target_from_source maps
source-camera coordinates into the target camera. Inputs must already be
undistorted/rectified with intrinsics for their *actual* crop and resolution.
No distortion, temporal compensation, hole filling, learned depth or calibration
refinement is silently applied. Numerical checks do not qualify real registration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUMERICAL_PIXEL_EPSILON = 1e-9  # Roundoff only, not a physical-error acceptance threshold.


def intrinsic_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0)
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(matrix[1, 0]) > 1e-12):
        raise ValueError("expected finite pinhole K with positive focal lengths")
    return matrix


def rigid_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError("expected finite homogeneous4x4 camera transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(rotation) - 1) > 1e-5):
        raise ValueError("rotation is not proper orthogonal; no automatic correction")
    return matrix


def from_millimetres(rotation: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation_mm, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape not in ((3,), (3, 1)):
        raise ValueError("expected3x3 rotation and3-vector translation")
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation.reshape(3) / 1000.
    return rigid_matrix(matrix)


def via_common_camera(target_from_common: np.ndarray,
                      source_from_common: np.ndarray) -> np.ndarray:
    """Compose target<-common<-source, e.g. RGB<-NIR<-thermal."""
    target = rigid_matrix(target_from_common)
    source = rigid_matrix(source_from_common)
    return rigid_matrix(target @ np.linalg.inv(source))


def cropped_resized_intrinsics(value: np.ndarray, crop_xy: tuple[float, float],
                               scale_xy: tuple[float, float]) -> np.ndarray:
    """Crop first, then resize with half-pixel pixel-centre correspondence.

    Never call this again if the provider's K already refers to the supplied crop.
    """
    matrix = intrinsic_matrix(value)
    crop = np.asarray(crop_xy, dtype=np.float64)
    scale = np.asarray(scale_xy, dtype=np.float64)
    if (crop.shape != (2,) or scale.shape != (2,) or not np.isfinite(crop).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("finite2D crop and positive2D scales required")
    adjustment = np.eye(3)
    adjustment[0, 0], adjustment[1, 1] = scale
    adjustment[:2, 2] = scale * (0.5 - crop) - 0.5
    return intrinsic_matrix(adjustment @ matrix)


def ms2_depth_metres(encoded: np.ndarray) -> np.ndarray:
    """Decode documented MS2 uint16 depth; NOT thermal sensor-count images.

    Missing (zero) depth remains NaN. There is no depth interpolation here.
    """
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or encoded.dtype != np.uint16:
        raise ValueError("MS2 depth input must be a2D uint16 array")
    result = encoded.astype(np.float64) / 256.
    result[encoded == 0] = np.nan
    return result


def image_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("image shape must be positive integer(height,width)")
    return shape


def pixel_support(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape(shape)
    return (np.isfinite(points).all(axis=1)
            & (points >= -NUMERICAL_PIXEL_EPSILON).all(axis=1)
            & (points <= np.array([width - 1, height - 1]) + NUMERICAL_PIXEL_EPSILON).all(axis=1))


@dataclass(frozen=True)
class Reprojection:
    target_xy: np.ndarray
    target_z_m: np.ndarray
    source_valid: np.ndarray
    target_in_front: np.ndarray
    target_in_bounds: np.ndarray

    @property
    def supported(self) -> np.ndarray:
        """Geometric support only: not independently confirmed visibility."""
        return self.source_valid & self.target_in_front & self.target_in_bounds


def project_pixels(source_xy: np.ndarray, source_z_m: np.ndarray,
                   source_k: np.ndarray, target_k: np.ndarray,
                   target_from_source: np.ndarray, *,
                   source_shape: tuple[int, int], target_shape: tuple[int, int]) -> Reprojection:
    source_k, target_k = intrinsic_matrix(source_k), intrinsic_matrix(target_k)
    transform = rigid_matrix(target_from_source)
    image_shape(source_shape)
    image_shape(target_shape)
    points = np.asarray(source_xy, dtype=np.float64)
    depth = np.asarray(source_z_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or depth.shape != (len(points),):
        raise ValueError("expected Nx2 pixels and N camera-axis depths")
    valid = pixel_support(points, source_shape) & np.isfinite(depth) & (depth > 0)
    xy, z = np.full_like(points, np.nan), np.full(len(points), np.nan)
    pixels_h = np.column_stack((points[valid], np.ones(np.count_nonzero(valid))))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera_source = (pixels_h @ np.linalg.inv(source_k).T) * depth[valid, None]
        camera_target = camera_source @ transform[:3, :3].T + transform[:3, 3]
        projected = camera_target @ target_k.T
        z[valid] = camera_target[:, 2]
        xy[valid] = projected[:, :2] / projected[:, 2, None]
    front = np.isfinite(z) & (z > 0)
    bounds = pixel_support(xy, target_shape)
    return Reprojection(xy, z, valid, front, bounds)


def zbuffer_candidates(projection: Reprojection, target_shape: tuple[int, int],
                       *, tie_tolerance_m: float = 1e-6) -> np.ndarray:
    """Nearest-centre z-buffer among projected samples; no unobserved occluder claim.

    Samples in the same target pixel behind the nearest observed Z are rejected.
    Missing depth elsewhere does not confirm visibility. Quantization and ties
    remain explicit and must not become independent correspondence ground truth.
    """
    height, width = image_shape(target_shape)
    if not np.isfinite(tie_tolerance_m) or tie_tolerance_m < 0:
        raise ValueError("nonnegative finite z-buffer tolerance required")
    valid = projection.supported.copy()
    # Recheck caller-supplied shape rather than trusting a previous image shape.
    valid &= pixel_support(projection.target_xy, target_shape)
    indices = np.flatnonzero(valid)
    coords = np.floor(projection.target_xy[indices] + 0.5).astype(np.int64)
    cells = coords[:, 1] * width + coords[:, 0]
    nearest = np.full(height * width, np.inf)
    np.minimum.at(nearest, cells, projection.target_z_m[indices])
    accepted = np.zeros(len(valid), dtype=bool)
    accepted[indices] = projection.target_z_m[indices] <= nearest[cells] + tie_tolerance_m
    return accepted
