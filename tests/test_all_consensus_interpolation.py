import numpy as np
import pytest

from scripts.probe_antiuav_all_consensus import local_interpolator


def test_affine_recovery_and_extrapolation_rejection():
    p = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 0], [0, 5]])
    warp = local_interpolator(p, p + 2, (20, 20))
    np.testing.assert_allclose(warp([[3, 4]]), [[5, 6]])
    assert np.isnan(warp([[20, 20]])).all()


def test_output_bounds():
    p = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 0], [0, 5]])
    warp = local_interpolator(p, p + 100, (20, 20))
    assert np.isnan(warp([[3, 4]])).all()


def test_insufficient_and_collinear():
    assert local_interpolator(np.empty((0, 2)), np.empty((0, 2)), (20, 20)) is None
    p = np.array([[i, i] for i in range(6)])
    assert local_interpolator(p, p, (20, 20)) is None
    with pytest.raises(ValueError):
        local_interpolator([[np.nan, 0]], [[0, 0]], (20, 20))
