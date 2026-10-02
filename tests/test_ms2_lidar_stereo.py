import numpy as np

from scripts.probe_ms2_lidar_stereo import aggregate, summarize_by_depth, timed_relative


def test_synchronous_rig_recovers_physical_baseline():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 1.0
    times = np.array([100, 200], np.int64)
    fixed, baseline = np.eye(4), np.eye(4)
    fixed[:3, 3] = [0.1, 0.3, 0.5]
    baseline[0, 3] = -0.3
    relative, info = timed_relative(poses, times, 150, 150, fixed, baseline)
    np.testing.assert_allclose(relative, baseline, atol=1e-12)
    assert info["available"]


def test_motion_uses_one_trajectory_and_correct_baseline_direction():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 1.0
    baseline = np.eye(4)
    baseline[0, 3] = -0.3
    relative, _ = timed_relative(
        poses, np.array([100, 200], np.int64), 100, 200, np.eye(4), baseline
    )
    assert relative[0, 3] == -1.3


def test_out_of_range_exposure_is_not_extrapolated():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    relative, info = timed_relative(
        poses, np.array([100, 200], np.int64), 99, 150, np.eye(4), np.eye(4)
    )
    assert relative is None and not info["available"]


def test_unavailable_frame_remains_in_aggregate_denominator():
    depth = np.array([5.0, 15.0, 30.0, 60.0])
    rows = [
        dict(sensor="thr", condition="paired_timed", strata=summarize_by_depth(np.zeros(4), depth)),
        dict(
            sensor="thr",
            condition="paired_timed",
            strata=summarize_by_depth(np.full(4, np.nan), depth),
        ),
    ]
    result = aggregate(rows)[0]
    assert result["reference_points"] == 8 and result["supported"] == 4
    assert result["planned_frames"] == 2 and result["frames_with_depth"] == 1
    assert result["equal_reference_frame_fraction_within_px"]["1.0"] == 0.5


def test_depth_strata_keep_missing_stereo_and_label_lidar_reference():
    result = summarize_by_depth(np.array([np.nan, 0.2]), np.array([5.0, 15.0]))
    assert result["lidar_depth_m_0_10"]["reference_points"] == 1
    assert result["lidar_depth_m_0_10"]["supported"] == 0
    assert result["lidar_depth_m_10_20"]["fraction_reference_within_px"]["1.0"] == 1
