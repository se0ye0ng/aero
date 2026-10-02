import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform


def trajectory():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 2
    times = np.array([1628215174000000000, 1628215175000000000], dtype=np.int64)
    return poses, times


def test_constant_velocity_direction_and_nanoseconds():
    poses, times = trajectory()
    at_target, info = interpolate_pose(poses, times, int(times[0]) + 250_000_000)
    assert at_target[0, 3] == 0.5
    assert info["alpha"] == 0.25
    assert info["extrapolated"] is False
    fixed = np.eye(4)
    fixed[1, 3] = 0.1
    result = moving_rig_transform(fixed, poses[0], at_target)
    np.testing.assert_allclose(result[:3, 3], [-0.5, 0.1, 0])
    np.testing.assert_allclose(moving_rig_transform(fixed, at_target, at_target), fixed)


def test_shortest_rotation_path():
    poses, times = trajectory()
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    result, _ = interpolate_pose(poses, times, int(times[0]) + 500_000_000)
    np.testing.assert_allclose(
        result[:3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix(), atol=1e-12
    )


def test_extrapolation_is_opt_in_and_recorded():
    poses, times = trajectory()
    query = int(times[0]) - 10_000_000
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_pose(poses, times, query)
    result, info = interpolate_pose(poses, times, query, max_extrapolation_ns=10_000_000)
    assert info["extrapolated"] is True
    assert info["extrapolation_ns"] == 10_000_000
    assert result[0, 3] == -0.02


def test_invalid_rotation_not_silently_repaired():
    poses, times = trajectory()
    poses[1, 0, 0] = 0.6
    with pytest.raises(ValueError, match="orthogonal"):
        interpolate_pose(poses, times, int(times[0]) + 100_000_000)


def test_exact_timestamp_preserves_input_and_identity():
    poses, times = trajectory()
    before = poses.copy()
    result, info = interpolate_pose(poses, times, int(times[0]))
    np.testing.assert_array_equal(result, poses[0])
    np.testing.assert_array_equal(poses, before)
    assert info["rotation_roundoff_projection_max_abs"] == 0


def test_nonincreasing_clock_rejected():
    poses, times = trajectory()
    with pytest.raises(ValueError, match="increasing"):
        interpolate_pose(poses, times[::-1], int(times[0]))
