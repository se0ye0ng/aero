"""Compose image-only coarse registration and residual matching in pixel centres."""

import numpy as np
from scipy.ndimage import map_coordinates


def field_predictor(field, source_size, target_size):
    """Normalized displacement, 2,H,W; return native-centre coordinates or NaN.

    No extrapolation outside the model's centre lattice or native target image.
    Source/target sizes are width,height, not the network's square input size.
    """
    f = np.asarray(field)
    if f.ndim != 3 or f.shape[0] != 2 or not np.isfinite(f).all():
        raise ValueError("finite 2,H,W field required")
    h, w = f.shape[1:]
    sizes = np.array([source_size, target_size], dtype=float)
    if sizes.shape != (2, 2) or not np.isfinite(sizes).all() or (sizes <= 1).any():
        raise ValueError("positive width,height dimensions greater than one required")

    def predict(points):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("Nx2 points required")
        out = np.full_like(p, np.nan)
        grid = (p + 0.5) / sizes[0] * [w, h] - 0.5
        valid = np.isfinite(grid).all(1) & (grid >= 0).all(1)
        valid &= (grid <= [w - 1, h - 1]).all(1)
        if valid.any():
            displacement = np.stack(
                [
                    map_coordinates(f[k], grid[valid, ::-1].T, order=1, prefilter=False)
                    for k in range(2)
                ],
                axis=1,
            )
            norm = (p[valid] + 0.5) / sizes[0] * 2 - 1 + displacement
            target = (norm + 1) / 2 * sizes[1] - 0.5
            inside = (target >= 0).all(1) & (target <= sizes[1] - 1).all(1)
            target[~inside] = np.nan
            out[valid] = target
        return out

    return predict


def observed(mask, points):
    p = np.asarray(points, dtype=float)
    valid = np.isfinite(p).all(1)
    valid &= (p >= 0).all(1) & (p <= np.array(mask.shape[::-1]) - 1).all(1)
    result = np.zeros(len(p), dtype=bool)
    if valid.any():
        # All bilinear support must be observed, not merely its nearest pixel.
        value = map_coordinates(
            np.asarray(mask, dtype=float), p[valid, ::-1].T, order=1, prefilter=False
        )
        result[valid] = value >= 1 - 1e-6
    return result


def compose(
    coarse_rgb_to_ir,
    coarse_ir_to_rgb,
    residual_forward,
    residual_reverse,
    support,
    *,
    rgb_top,
    ir_top,
):
    """Residual maps use uncropped IR coordinates in both directions.

    Warped RGB's coordinate is an IR coordinate, not an original RGB coordinate.
    Invalid rendered pixels cannot become evidence through residual matching.
    """

    def guarded(predict, points):
        out = np.full_like(points, np.nan, dtype=float)
        finite = np.isfinite(points).all(1)
        if finite.any():
            out[finite] = predict(points[finite])
        return out

    def forward(points):
        points = np.asarray(points, dtype=float)
        common = guarded(coarse_rgb_to_ir, points)
        out = guarded(residual_forward, common)
        valid = observed(support, common) & (points[:, 1] >= rgb_top)
        valid &= (common[:, 1] >= ir_top) & (out[:, 1] >= ir_top)
        out[~valid] = np.nan
        return out

    def reverse(points):
        points = np.asarray(points, dtype=float)
        common = guarded(residual_reverse, points)
        out = guarded(coarse_ir_to_rgb, common)
        valid = observed(support, common) & (points[:, 1] >= ir_top)
        valid &= (common[:, 1] >= ir_top) & (out[:, 1] >= rgb_top)
        out[~valid] = np.nan
        return out

    return forward, reverse
