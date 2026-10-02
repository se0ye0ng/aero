import numpy as np
import pytest

from aero_ir.registration.stereo_vertical_compensation import estimate, shift_right
from aero_ir.registration.temporal_stereo import estimate_pair, rectification


def test_zero_sampling_is_exact_and_does_not_alias_inputs():
    image = np.arange(40, dtype=np.uint8).reshape(8, 5)
    mask = np.ones(image.shape, bool)
    actual, support = shift_right(image, mask, 0.0)
    np.testing.assert_array_equal(actual, image)
    np.testing.assert_array_equal(support, mask)
    actual[0, 0] = 255
    assert image[0, 0] == 0


def test_half_pixel_sign_and_observed_contributors():
    image = np.repeat((np.arange(8) * 10).astype(np.uint8)[:, None], 5, axis=1)
    mask = np.ones(image.shape, bool)
    mask[3] = False
    plus, mp = shift_right(image, mask, 0.5)
    minus, mm = shift_right(image, mask, -0.5)
    assert plus[1, 2] == 15 and minus[1, 2] == 5
    assert not mp[-1].any() and not mm[0].any()
    assert not mp[2:4].any() and not mm[3:5].any()


def test_zero_estimator_reproduces_existing_rectified_pipeline():
    left = np.random.default_rng(0).integers(0, 256, (96, 320), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]
    k = np.array([[150.0, 0.0, 160.0], [0.0, 150.0, 48.0], [0.0, 0.0, 1.0]])
    b = np.eye(4)
    b[0, 3] = -0.3
    g = rectification(k, k, b, left.shape)
    expected = estimate_pair(left, right, g)
    actual = estimate(left, right, g, 0.0)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_unplanned_offset_rejected():
    with pytest.raises(ValueError, match="half-pixel"):
        shift_right(np.ones((8, 8), np.uint8), np.ones((8, 8), bool), 1.0)
