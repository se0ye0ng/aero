import numpy as np

from aero_ir.registration.temporal_stereo import points_from_disparity, rectification
from scripts.analyze_ms2_cross_stereo import expected_disparity, strata, summarize_errors


def test_expected_disparity_and_native_depth_reconstruction_are_inverse_under_rotation():
    k = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 48.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    angle = 0.03
    transform[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    transform[:3, 3] = [-0.3, 0.02, 0.01]
    geometry = rectification(k, k, transform, (96, 320))
    xy, z = np.array([[170.0, 40.0], [140.0, 50.0]]), np.array([10.0, 30.0])
    predicted = expected_disparity(xy, z, geometry)
    reconstructed = points_from_disparity(xy, predicted, geometry)
    np.testing.assert_allclose(reconstructed[:, 2], z, atol=1e-10)


def test_equal_disparity_error_does_not_mean_equal_relative_depth_error():
    # f*baseline60px*m: adding1disparity pixel has different range effects.
    truth = np.array([10.0, 60.0])
    disparity = 60 / truth
    estimated = 60 / (disparity + 1.0)
    relative = abs(estimated - truth) / truth
    assert relative[0] < 0.15 and relative[1] == 0.5


def test_invalid_points_stay_in_full_reference_disparity_denominator():
    result = summarize_errors(np.array([0.2, 2.0, np.nan]), np.array([True, True, True]))
    assert result["reference_points"] == 3 and result["supported"] == 2
    assert result["fraction_reference_within_px"]["1.0"] == 1 / 3
    assert result["fraction_available_within_px"]["1.0"] == 0.5
    assert summarize_errors(np.zeros(0), np.zeros(0, bool))["conditional_median_abs_px"] is None


def test_fixed_source_depth_bins_partition_without_camera_dependent_selection():
    masks = strata(np.array([1.0, 10.0, 20.0, 40.0]))
    np.testing.assert_array_equal(
        sum(v.astype(int) for k, v in masks.items() if k != "all"), np.ones(4)
    )
