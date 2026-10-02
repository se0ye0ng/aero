import numpy as np

from scripts.probe_external_orientation_repair import prune_controls, selected_tps


def controls():
    y, x = np.mgrid[0:3, 0:3]
    return np.c_[x.ravel(), y.ravel()].astype(float)


def test_positive_affine_controls_unchanged():
    x = controls()
    keep, history, status = prune_controls(x, x * 2 + 1, np.ones(len(x)))
    np.testing.assert_array_equal(keep, np.arange(9))
    assert not history and status == "positive_control_triangles"


def test_global_reflection_not_declared_repaired():
    x = controls()
    keep, history, status = prune_controls(x, x * [-1, 1], np.ones(len(x)))
    assert status != "positive_control_triangles"
    assert len(keep) >= 6
    assert len(history) == 9 - len(keep)


def test_pruning_repeatable_and_reference_free():
    x = controls()
    target = x.copy()
    target[4] = [4, 4]
    first = prune_controls(x, target, np.linspace(0, 1, 9))
    second = prune_controls(x, target, np.linspace(0, 1, 9))
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_selected_tps_affine_and_unsupported():
    x = controls() * 10
    warp = selected_tps(x, x + 5, (40, 40), (40, 40), 0.0001)
    result = warp([[5, 5], [35, 35]])
    np.testing.assert_allclose(result[0], [10, 10], atol=1e-10)
    assert np.isnan(result[1]).all()
