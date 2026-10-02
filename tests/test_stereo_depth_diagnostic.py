import numpy as np
import pytest

from aero_ir.registration.stereo_depth_diagnostic import (
    SGBM_SETTINGS,
    compute_stereo,
    sample_disparity,
)


def test_known_disparity_sign_scale_and_reverse():
    rng = np.random.default_rng(41)
    left = rng.integers(0, 256, (96, 320), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]
    result = compute_stereo(left, right, settings=SGBM_SETTINGS | dict(numDisparities=64))
    region = result["valid"][16:-16, 90:230]
    assert region.mean() > 0.9
    assert abs(np.median(result["disparity"][16:-16, 90:230][region]) - 8) < 0.1


def test_sampling_rejects_holes_edges_and_outside():
    disparity = np.full((5, 6), 8.0)
    valid = np.ones_like(disparity, dtype=bool)
    valid[2, 2] = False
    disparity[0, 5] = 12
    values, mask = sample_disparity(
        dict(disparity=disparity, valid=valid), [[0.5, 0.5], [1.5, 1.5], [4.5, 0.2], [-0.1, 1]]
    )
    np.testing.assert_array_equal(mask, [True, False, False, False])
    assert values[0] == 8
    assert np.isnan(values[1:]).all()


def test_dtype_is_explicit():
    with pytest.raises(ValueError):
        compute_stereo(np.zeros((50, 160)), np.zeros((50, 160)))
