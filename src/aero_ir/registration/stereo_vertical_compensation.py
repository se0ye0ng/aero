"""Exploratory rectified-image sampling intervention, not physical recalibration."""

import cv2
import numpy as np

from aero_ir.registration.raft_stereo import rectify_images
from aero_ir.registration.stereo_depth_diagnostic import compute_stereo


def shift_right(image, support, offset):
    if offset not in (0.0, 0.5, -0.5) or image.ndim != 2 or support.shape != image.shape:
        raise ValueError("fixed zero/plus/minus half-pixel offset and equal 2D arrays required")
    if image.dtype != np.uint8 or support.dtype != bool:
        raise ValueError("uint8 image and boolean observation mask required")
    if offset == 0:
        return image.copy(), support.copy()
    h, w = image.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    yy += offset
    result = cv2.remap(
        image, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    lo = np.floor(yy).astype(int)
    inside = (lo >= 0) & (lo + 1 < h)
    clipped = np.clip(lo, 0, h - 2)
    observed = inside & support[clipped, xx.astype(int)] & support[clipped + 1, xx.astype(int)]
    return result, observed


def estimate(left, right, geometry, offset):
    images, supports = rectify_images(left, right, geometry)
    images[1], supports[1] = shift_right(images[1], supports[1], offset)
    stereo = compute_stereo(*images)
    h, w = images[0].shape
    yy, xx = np.indices((h, w))
    x0 = np.floor(xx - stereo["disparity"]).astype(int)
    clipped = np.clip(x0, 0, w - 2)
    stereo["valid"] &= (
        supports[0]
        & (x0 >= 0)
        & (x0 + 1 < w)
        & supports[1][yy, clipped]
        & supports[1][yy, clipped + 1]
    )
    stereo.update(rectified_left_support=supports[0], rectified_right_support=supports[1])
    return stereo
