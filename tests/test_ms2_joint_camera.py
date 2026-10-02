import numpy as np
import pytest

from scripts.probe_ms2_joint_camera import fit_joint, joint_predict


def data():
    rng = np.random.default_rng(19)
    points = rng.uniform([-10.0, -5.0, 12.0], [10.0, 5.0, 35.0], (120, 3))
    k = np.array([[390.0, 0.0, 300.0], [0.0, 385.0, 150.0], [0.0, 0.0, 1.0]])
    return points, k


def test_joint_known_parameters_recovered():
    points, k = data()
    truth = np.array([0.04, -0.08, 0.03, -0.3, 0.2, 0.1, -0.2])
    target = joint_predict(points, k, truth)
    fit = fit_joint(points, target, k, np.zeros(7))
    assert fit["success"]
    assert not any(fit["active_bounds"])
    np.testing.assert_allclose(fit["parameters"], truth, atol=1e-5)
    np.testing.assert_allclose(joint_predict(points, k, fit["parameters"]), target, atol=1e-6)


@pytest.mark.parametrize("parameters", [np.zeros(6), np.full(7, np.nan), np.full(7, 1.01)])
def test_joint_rejects_invalid_parameters(parameters):
    points, k = data()
    with pytest.raises(ValueError):
        joint_predict(points, k, parameters)


def test_joint_rejects_behind_camera_points():
    points, k = data()
    target = joint_predict(points, k, np.zeros(7))
    points[0, 2] = -1.0
    with pytest.raises(ValueError):
        fit_joint(points, target, k, np.zeros(7))
