import numpy as np
import pytest

from scripts.probe_ms2_resize_control import residual_summary, resize_correspondence


def test_pixel_centres_and_inverse():
    points = np.array([[0.0, 0.0], [100.0, 200.0], [1223.0, 383.0]])
    target = resize_correspondence(points, [1224, 384], [640, 256])
    np.testing.assert_allclose(target[0], [0.5 * 640 / 1224 - 0.5, 0.5 * 256 / 384 - 0.5])
    np.testing.assert_allclose(
        resize_correspondence(target, [640, 256], [1224, 384]), points, atol=1e-12
    )


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        resize_correspondence([[0, 0]], [0, 384], [640, 256])


def test_residuals_keep_large_errors():
    result = residual_summary(np.array([[0.0, 0.0], [0.0, 100.0]]))
    assert result["matches"] == 2
    assert result["within_target_px_counts"]["3.0"] == 1
    assert result["median_signed_xy"] == [0.0, 50.0]
    assert residual_summary(np.empty((0, 2)))["median_target_px"] is None
