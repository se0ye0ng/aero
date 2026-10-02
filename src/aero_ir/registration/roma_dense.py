"""Query symmetric RoMa output without sparse sampling or TPS refitting."""

import numpy as np
from scipy.ndimage import map_coordinates

from aero_ir.registration.roma_coordinates import normalized_to_centres


def split_dense(warp, certainty):
    w, c = np.asarray(warp), np.asarray(certainty)
    if w.ndim != 3 or w.shape[2] != 4 or w.shape[1] % 2 or c.shape != w.shape[:2]:
        raise ValueError("symmetric Hx2Wx4 warp and Hx2W certainty required")
    if not np.isfinite(w).all() or not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
        raise ValueError("finite warp and probability certainty required")
    h, width = w.shape[0], w.shape[1] // 2
    xx, yy = np.meshgrid(2 * (np.arange(width) + 0.5) / width - 1, 2 * (np.arange(h) + 0.5) / h - 1)
    grid = np.stack([xx, yy], axis=-1)
    np.testing.assert_allclose(w[:, :width, :2], grid, atol=2e-6, rtol=0)
    np.testing.assert_allclose(w[:, width:, 2:], grid, atol=2e-6, rtol=0)
    return (w[:, :width, 2:], c[:, :width]), (w[:, width:, :2], c[:, width:])


def dense_predictor(field, certainty, source_size, target_size, source_top=0, target_top=0):
    """Fixed confidence >=0.5; in-frame target support; input/output base pixel centres."""
    field, certainty = np.asarray(field), np.asarray(certainty)
    h, w = certainty.shape

    def predict(points):
        p = np.asarray(points, dtype=float) - [0, source_top]
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError("finite Nx2 queries required")
        inside = ((p >= 0) & (p <= np.asarray(source_size) - 1)).all(1)
        sample = (p + 0.5) / source_size * [w, h] - 0.5
        norm = np.stack(
            [
                map_coordinates(
                    field[..., k], sample[:, ::-1].T, order=1, mode="nearest", prefilter=False
                )
                for k in range(2)
            ],
            axis=1,
        )
        conf = map_coordinates(
            certainty, sample[:, ::-1].T, order=1, mode="nearest", prefilter=False
        )
        target = normalized_to_centres(norm, target_size)
        inside &= ((target >= 0) & (target <= np.asarray(target_size) - 1)).all(1)
        inside &= conf >= 0.5
        out = target + [0, target_top]
        out[~inside] = np.nan
        return out

    return predict
