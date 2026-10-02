import numpy as np

from scripts.probe_external_interpolation import linear_warp


def test_linear_warp_affine_exact_and_no_extrapolation():
    controls = np.array([[10.0, 10.0], [80.0, 10.0], [10.0, 80.0], [80.0, 80.0]])
    warp = linear_warp(controls, controls * 0.5 + 3, (100, 100), (60, 60))
    points = np.array([[20.0, 20.0], [60.0, 40.0], [0.0, 0.0]])
    result = warp(points)
    np.testing.assert_allclose(result[:2], points[:2] * 0.5 + 3)
    assert np.isnan(result[2]).all()


def test_linear_warp_rejects_out_of_target_not_clipped():
    controls = np.array([[0.0, 0.0], [90.0, 0.0], [0.0, 90.0]])
    warp = linear_warp(controls, controls + 20, (100, 100), (30, 30))
    result = warp(np.array([[5.0, 5.0], [30.0, 30.0]]))
    np.testing.assert_allclose(result[0], [25, 25])
    assert np.isnan(result[1]).all()
