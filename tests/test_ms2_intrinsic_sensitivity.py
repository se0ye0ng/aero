import numpy as np
import pytest

from scripts.probe_ms2_intrinsic_sensitivity import corrected_intrinsics, fit_intrinsics


def test_small_known_intrinsic_perturbation_is_recovered_without_pose_fit():
    rng = np.random.default_rng(4)
    points = rng.uniform([-4.0, -2.0, 5.0], [4.0, 2.0, 20.0], (100, 3))
    k = np.array([[400.0, 0.0, 300.0], [0.0, 400.0, 150.0], [0.0, 0.0, 1.0]])
    true = np.array([0.3, -0.5, 0.2, -0.4])
    observed = points @ corrected_intrinsics(k, true).T
    observed = observed[:, :2] / observed[:, 2, None]
    result = fit_intrinsics(points, observed, k)
    assert result["optimizer_success"] and not any(result["bound_hit"])
    np.testing.assert_allclose(result["parameters"], true, atol=1e-6)


def test_fitting_does_not_widen_bounds_to_follow_large_offset():
    rng = np.random.default_rng(3)
    points = rng.uniform([-1.0, -1.0, 5.0], [1.0, 1.0, 10.0], (20, 3))
    k = np.diag([100.0, 100.0, 1.0])
    xy = points @ k.T
    observed = xy[:, :2] / xy[:, 2, None] + 20.0
    fit = fit_intrinsics(points, observed, k)
    assert fit["bound_hit"][2:] == [True, True]
    assert np.max(np.abs(fit["parameters"])) <= 1.0


def test_invalid_fit_points_rejected_instead_of_silently_dropped():
    with pytest.raises(ValueError):
        fit_intrinsics(np.full((10, 3), np.nan), np.ones((10, 2)), np.eye(3))
