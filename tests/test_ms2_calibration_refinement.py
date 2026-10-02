import numpy as np
import pytest

from scripts.probe_ms2_calibration_refinement import correction, fit, predict


def test_recover_synthetic_rigid_correction():
    rng = np.random.default_rng(17)
    points = rng.uniform([-3, -2, 5], [3, 2, 30], size=(80, 3))
    k = np.array([[400.0, 0, 320], [0, 400.0, 128], [0, 0, 1]])
    transform = correction([0.1, -0.2, 0.15, 0.2, -0.1, 0.3])
    target = predict(points, k, transform)
    result = fit(points, target, k, 6)
    assert result["success"]
    assert result["normalized_jacobian_rank"] == 6
    np.testing.assert_allclose(result["transform"], transform, atol=1e-7)


def test_reject_behind_camera_and_bad_parameters():
    with pytest.raises(ValueError):
        predict(np.array([[0, 0, -1.0]]), np.eye(3), np.eye(4))
    with pytest.raises(ValueError):
        correction([0, 0])
