"""Stereo rectification for an explicit inter-exposure camera transform.

Inputs are already lens-undistorted native images. This module applies only the
specified pose geometry; it does not estimate timing, optical distortion or
independently moving objects. Estimated depth is not correspondence ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np

from aero_ir.registration.calibrated import intrinsic_matrix, pixel_support, rigid_matrix
from aero_ir.registration.stereo_depth_diagnostic import compute_stereo, sample_disparity


def rectification(k_left, k_right, right_from_left, shape):
    left, right = intrinsic_matrix(k_left), intrinsic_matrix(k_right)
    transform = rigid_matrix(right_from_left)
    h, w = shape
    if h < 8 or w < 160 or np.linalg.norm(transform[:3, 3]) < 1e-6:
        raise ValueError('nonzero stereo baseline and sufficiently large image required')
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        left, np.zeros((5, 1)), right, np.zeros((5, 1)), (w, h),
        np.ascontiguousarray(transform[:3, :3]), np.ascontiguousarray(transform[:3, 3:4]),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1., newImageSize=(w, h))
    if (p2[0, 3] >= -1e-9 or abs(p2[1, 3]) > 1e-9
            or not all(np.isfinite(m).all() for m in (r1, r2, p1, p2, q))):
        raise ValueError('expected horizontal positive-left-disparity rectification')
    return dict(left_k=left, right_k=right, right_from_left=transform, shape=np.array(shape),
                r1=r1, r2=r2, p1=p1, p2=p2, q=q)


def native_to_rectified(points, geometry):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('finite Nx2 native image points required')
    rays = np.c_[points, np.ones(len(points))] @ np.linalg.inv(geometry['left_k']).T
    rectified = rays @ geometry['r1'].T @ geometry['p1'][:, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        return rectified[:, :2]/rectified[:, 2, None]


def points_from_disparity(native_points, disparity, geometry):
    """Reconstruct in the ORIGINAL left-camera coordinate system, not rectified Z."""
    disparity = np.asarray(disparity, dtype=np.float64)
    rectified = native_to_rectified(native_points, geometry)
    if disparity.shape != (len(rectified),):
        raise ValueError('one disparity per native point required')
    homogeneous = np.c_[rectified, disparity, np.ones(len(rectified))] @ geometry['q'].T
    valid = (np.isfinite(homogeneous).all(axis=1) & np.isfinite(disparity) & (disparity > 0)
             & (np.abs(homogeneous[:, 3]) > 1e-12))
    points = np.full((len(rectified), 3), np.nan)
    points[valid] = (homogeneous[valid, :3]/homogeneous[valid, 3, None]) @ geometry['r1']
    points[points[:, 2] <= 0] = np.nan
    return points


def estimate_pair(left, right, geometry):
    """Rectify, mask remapping borders, then use the frozen RGB-stereo settings."""
    shape = tuple(int(v) for v in geometry['shape'])
    if (left.dtype != np.uint8 or right.dtype != np.uint8
            or left.shape != shape or right.shape != shape):
        raise ValueError('equal native uint8 grayscale images required')
    h, w = shape
    rectified, supports = [], []
    for image, k, r, p in ((left, geometry['left_k'], geometry['r1'], geometry['p1']),
                            (right, geometry['right_k'], geometry['r2'], geometry['p2'])):
        mx, my = cv2.initUndistortRectifyMap(k, np.zeros(5), r, p[:, :3], (w, h), cv2.CV_32FC1)
        support = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (my >= 0)
        support &= (mx < w-1) & (my < h-1)
        # BlockSize5 requires a 2-pixel valid footprint, not just a valid centre.
        support = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        supports.append(support)
        rectified.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    stereo = compute_stereo(*rectified)
    yy, xx = np.indices(shape)
    right_x = xx-stereo['disparity']
    x0 = np.floor(right_x).astype(np.int64)
    clipped = np.clip(x0, 0, w-2)
    stereo['valid'] &= (supports[0] & (x0 >= 0) & (x0+1 < w)
                        & supports[1][yy, clipped] & supports[1][yy, clipped+1])
    stereo.update(rectified_left_support=supports[0], rectified_right_support=supports[1])
    return stereo


def sample_native_depth(stereo, geometry, native_points):
    rectified = native_to_rectified(native_points, geometry)
    finite = np.isfinite(rectified).all(axis=1)
    disparity = np.full(len(rectified), np.nan)
    disparity[finite], _ = sample_disparity(stereo, rectified[finite])
    points = points_from_disparity(native_points, disparity, geometry)
    valid = pixel_support(np.asarray(native_points), tuple(int(v) for v in geometry['shape']))
    valid &= np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    depth = points[:, 2].copy()
    depth[~valid] = np.nan
    return depth, valid
