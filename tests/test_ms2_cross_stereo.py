import numpy as np

from scripts.probe_ms2_cross_stereo import depth_statistics


def test_missing_thermal_depth_remains_failure_in_reference_population():
    scores, error, valid = depth_statistics(
        [10.0, 20.0, 30.0], [10.0, np.nan, 0.0], np.array([True, True, True])
    )
    assert scores["reference_points"] == 3 and scores["thermal_depth_available"] == 1
    assert scores["fraction_reference_within_10pct"] == 1 / 3
    assert scores["fraction_available_within_10pct"] == 1
    np.testing.assert_array_equal(valid, [True, False, False])
    assert np.isinf(error[1:]).all()


def test_depth_error_uses_expected_native_camera_z_and_excludes_only_fixed_reference():
    scores, _, _ = depth_statistics(
        [10.0, 20.0, 100.0], [11.0, 24.0, 100.0], np.array([True, True, False])
    )
    assert scores["fraction_reference_within_10pct"] == 0.5
    assert np.isclose(scores["conditional_median_relative_error"], 0.15)
    empty, _, _ = depth_statistics([], [], np.array([], bool))
    assert empty["fraction_reference_within_10pct"] is None
