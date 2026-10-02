import numpy as np

from aero_ir.registration.temporal_stereo import rectification
from scripts.probe_ms2_raft_stereo import score


def test_unavailable_pose_keeps_all_lidar_points_as_failures():
    original = dict(
        source_xy=np.array([[10.0, 10.0], [20.0, 20.0]]),
        lidar_depth_m=np.array([10.0, 20.0]),
        equivalent_disparity_errors=np.full(2, np.nan),
    )
    z, errors, result = score({}, original, 200.0)
    assert np.isnan(z).all() and np.isnan(errors).all()
    assert result["strata"]["all"]["reference_points"] == 2
    assert result["strata"]["all"]["fraction_reference_within_px"]["1.0"] == 0
    assert result["common_support"]["raft"]["reference_points"] == 0


def test_known_stereo_depth_and_fixed_baseline_common_support():
    k = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 48.0], [0.0, 0.0, 1.0]])
    b = np.eye(4)
    b[0, 3] = -0.3
    geometry = rectification(k, k, b, (96, 320))
    d = -geometry["p2"][0, 3] / 10.0
    original = dict(
        source_xy=np.array([[170.0, 40.0], [180.0, 40.0]]),
        lidar_depth_m=np.array([10.0, 10.0]),
        equivalent_disparity_errors=np.array([2.0, np.nan]),
        **{f"geometry_{key}": value for key, value in geometry.items()},
    )
    stereo = dict(disparity=np.full((96, 320), d), valid=np.ones((96, 320), bool))
    z, errors, result = score(stereo, original, 200.0)
    np.testing.assert_allclose(z, 10.0, atol=1e-10)
    np.testing.assert_allclose(errors, 0.0, atol=1e-10)
    assert result["strata"]["all"]["supported"] == 2
    assert result["common_support"]["raft"]["reference_points"] == 1
    assert result["common_support"]["baseline"]["conditional_median_abs_px"] == 2.0


def test_inaccurate_available_prediction_is_not_filtered_using_lidar_error():
    k = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 48.0], [0.0, 0.0, 1.0]])
    b = np.eye(4)
    b[0, 3] = -0.3
    geometry = rectification(k, k, b, (96, 320))
    original = dict(
        source_xy=np.array([[170.0, 40.0]]),
        lidar_depth_m=np.array([10.0]),
        equivalent_disparity_errors=np.array([0.1]),
        **{f"geometry_{key}": value for key, value in geometry.items()},
    )
    stereo = dict(disparity=np.full((96, 320), 20.0), valid=np.ones((96, 320), bool))
    _, errors, result = score(stereo, original, 200.0)
    assert errors[0] > 10.0
    assert result["strata"]["all"]["supported"] == 1
    assert result["strata"]["all"]["fraction_reference_within_px"]["3.0"] == 0
