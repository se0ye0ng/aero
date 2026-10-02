"""Independent RGB stereo diagnostic. Estimated depth is never pixel GT."""
from __future__ import annotations

import cv2
import numpy as np

SGBM_SETTINGS = dict(minDisparity=0, numDisparities=128, blockSize=5,
                     P1=8*25, P2=32*25, disp12MaxDiff=1, preFilterCap=31,
                     uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                     mode=cv2.STEREO_SGBM_MODE_SGBM)


def compute_stereo(left, right, *, settings=None):
    """Left-positive disparity and explicit left/right consistency, no hole filling."""
    if (left.dtype != np.uint8 or right.dtype != np.uint8 or left.ndim != 2
            or left.shape != right.shape):
        raise ValueError("equal-sized uint8 grayscale images required")
    settings = SGBM_SETTINGS if settings is None else settings
    matcher = cv2.StereoSGBM_create(**settings)
    disparity = matcher.compute(left, right).astype(np.float64)/16.
    # Mirror both views and swap them: the same positive-disparity convention then applies.
    reverse = matcher.compute(np.ascontiguousarray(right[:, ::-1]),
                              np.ascontiguousarray(left[:, ::-1])).astype(np.float64)/16.
    reverse = reverse[:, ::-1]
    h, w = left.shape
    yy, xx = np.indices((h, w))
    xr = xx - disparity
    x0 = np.floor(xr).astype(np.int64)
    in_bounds = (x0 >= 0) & (x0+1 < w)
    clipped = np.clip(x0, 0, w-2)
    r0, r1 = reverse[yy, clipped], reverse[yy, clipped+1]
    reverse_valid = (r0 > 0) & (r1 > 0)
    blend = xr-x0
    sampled = r0*(1-blend) + r1*blend
    lr_error = np.abs(disparity-sampled)
    valid = (disparity > 0) & in_bounds & reverse_valid & (lr_error <= 1.)
    return dict(disparity=disparity, reverse_disparity=reverse,
                valid=valid, lr_error=lr_error)


def sample_disparity(stereo, points):
    """Conservative four-neighbor sample; discontinuities/missing support stay invalid."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("finite Nx2 pixel coordinates required")
    disparity, mask = stereo["disparity"], stereo["valid"]
    h, w = disparity.shape
    ij = np.floor(points).astype(np.int64)
    x, y = ij.T
    in_bounds = (x >= 0) & (x+1 < w) & (y >= 0) & (y+1 < h)
    x, y = np.clip(x, 0, w-2), np.clip(y, 0, h-2)
    values = np.stack([disparity[y, x], disparity[y, x+1],
                       disparity[y+1, x], disparity[y+1, x+1]], axis=1)
    supported = np.stack([mask[y, x], mask[y, x+1], mask[y+1, x], mask[y+1, x+1]], axis=1)
    valid = (in_bounds & supported.all(1) & np.isfinite(values).all(1)
             & (values > 0).all(1) & (np.ptp(values, axis=1) <= 1.))
    dx, dy = (points-ij).T
    weights = np.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], axis=1)
    sampled = np.sum(values*weights, axis=1)
    sampled[~valid] = np.nan
    return sampled, valid
