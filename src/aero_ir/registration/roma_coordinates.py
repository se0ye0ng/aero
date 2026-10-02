"""RoMa align_corners=False grids to AERO zero-based pixel-centre coordinates."""

import numpy as np


def normalized_to_centres(points, size_wh, header_top=0):
    """Map cropped normalized coordinates; never clamp out-of-support predictions.

    RoMa's upstream convenience conversion yields edge-origin coordinates. Its
    source grid is linspace(-1+1/W, 1-1/W, W), so subtracting 0.5 maps it onto
    integer pixel centres used by the existing AERO matching and crop contract.
    """
    p = np.asarray(points, dtype=float)
    size = np.asarray(size_wh, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("finite Nx2 normalized coordinates required")
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("positive width and height required")
    if not np.isfinite(header_top) or header_top < 0:
        raise ValueError("nonnegative finite header offset required")
    return (p + 1) * size / 2 - 0.5 + [0, header_top]


def rgb_image(array):
    """Explicit RGB or replicated grayscale input; no implicit BGR interpretation."""
    from PIL import Image

    a = np.asarray(array)
    if a.dtype != np.uint8:
        raise ValueError("uint8 image required")
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 1:
        raise ValueError("nonempty grayscale or RGB image required")
    return Image.fromarray(a)
