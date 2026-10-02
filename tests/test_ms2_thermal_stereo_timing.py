import numpy as np

from aero_ir.registration.calibrated import project_pixels
from aero_ir.registration.ego_motion import moving_rig_transform
from scripts.probe_ms2_thermal_stereo_timing import statistics, stereo_transforms


def test_known_inter_camera_motion_changes_disparity_and_vertical_alignment():
    k = np.diag([100.0, 100.0, 1.0])
    baseline, pose = np.eye(4), np.eye(4)
    baseline[0, 3] = -0.3
    pose[:2, 3] = [0.08, 0.02]
    right_time = moving_rig_transform(np.eye(4), np.eye(4), pose)
    transforms = stereo_transforms(np.eye(4), right_time, baseline, np.eye(4))
    xy = {}
    for name, transform in transforms.items():
        xy[name] = project_pixels(
            np.array([[10.0, 10.0]]),
            np.array([10.0]),
            k,
            k,
            transform,
            source_shape=(32, 32),
            target_shape=(32, 32),
        ).target_xy
    np.testing.assert_allclose(xy["left"] - xy["simultaneous_right"], [[3.0, 0.0]])
    np.testing.assert_allclose(xy["actual_right"] - xy["simultaneous_right"], [[-0.8, -0.2]])


def test_same_time_preserves_stereo_baseline_after_camera_rotation():
    angle = 0.05
    correction = np.eye(4)
    correction[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    baseline = np.eye(4)
    baseline[0, 3] = -0.3
    transforms = stereo_transforms(np.eye(4), np.eye(4), baseline, correction)
    np.testing.assert_allclose(transforms["actual_right"], transforms["simultaneous_right"])
    np.testing.assert_allclose(
        transforms["actual_right"] @ np.linalg.inv(transforms["left"]), baseline, atol=1e-12
    )


def test_statistics_retains_invalid_count_and_empty_is_not_zero_error():
    result = statistics(np.array([1.0, 2.0, np.nan, np.inf]))
    assert result["count"] == 4 and result["finite"] == 2 and result["median"] == 1.5
    assert statistics([])["median"] is None
