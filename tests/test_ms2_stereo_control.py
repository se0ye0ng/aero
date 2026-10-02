import numpy as np

from aero_ir.registration.ego_motion import moving_rig_transform
from scripts.probe_ms2_stereo_control import camera_pose


def test_camera_pose_and_no_motion_stereo():
    pose = np.eye(4)
    pose[:3, 3] = [10, 3, 2]
    sensor_from_rgb = np.eye(4)
    sensor_from_rgb[1, 3] = 0.3
    camera = camera_pose(pose, sensor_from_rgb)
    np.testing.assert_allclose(camera @ sensor_from_rgb, pose)
    stereo = np.eye(4)
    stereo[0, 3] = -0.3
    np.testing.assert_allclose(moving_rig_transform(stereo, camera, camera), stereo, atol=1e-12)


def test_rotating_rig_matches_world_coordinate_chain():
    angle = 0.2
    sensor = np.eye(4)
    sensor[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    sensor[:3, 3] = [0.01, 0.3, 0.5]
    before = np.eye(4)
    before[:3, 3] = [10, 2, 3]
    after = before.copy()
    after[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    after[:3, 3] += [0.1, 0.2, 0.3]
    stereo = np.eye(4)
    stereo[0, 3] = -0.3
    transform = moving_rig_transform(
        stereo, camera_pose(before, sensor), camera_pose(after, sensor)
    )
    source_point = np.array([1, 2, 15, 1.0])
    world_point = before @ np.linalg.solve(sensor, source_point)
    expected = stereo @ sensor @ np.linalg.solve(after, world_point)
    np.testing.assert_allclose(transform @ source_point, expected, atol=1e-12)
