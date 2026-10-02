"""Explicit world-from-camera pose interpolation for a moving rigid sensor rig.

This is a motion-model hypothesis, not proof of pose/exposure accuracy. Independently
moving objects, rolling shutter and per-return LiDAR timing are not compensated.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from aero_ir.registration.calibrated import rigid_matrix


def interpolate_pose(poses: np.ndarray, timestamps_ns: np.ndarray, query_ns: int,
                     *, max_extrapolation_ns: int = 0) -> tuple[np.ndarray, dict]:
    poses = np.asarray(poses, dtype=np.float64)
    times = np.asarray(timestamps_ns)
    if (times.ndim != 1 or times.dtype != np.int64 or len(times) < 2
            or poses.shape != (len(times), 4, 4) or (times <= 0).any()
            or (np.diff(times) <= 0).any() or type(query_ns) is not int or query_ns <= 0
            or type(max_extrapolation_ns) is not int or max_extrapolation_ns < 0):
        raise ValueError("expected aligned poses and increasing positive int64 timestamps")
    outside = max(int(times[0]) - query_ns, query_ns - int(times[-1]), 0)
    if outside > max_extrapolation_ns:
        raise ValueError("query requires extrapolation beyond the explicit bound")
    lower = int(np.clip(np.searchsorted(times, query_ns, side="right") - 1, 0, len(times) - 2))
    upper = lower + 1
    start, end = rigid_matrix(poses[lower]), rigid_matrix(poses[upper])
    interval = int(times[upper]) - int(times[lower])
    alpha = (query_ns - int(times[lower])) / interval
    info = dict(lower_index=lower, upper_index=upper, alpha=alpha, bracket_ns=interval,
                extrapolated=outside > 0, extrapolation_ns=outside,
                max_extrapolation_ns=max_extrapolation_ns,
                rotation_roundoff_projection_max_abs=0.)
    if alpha in (0., 1.):
        return (start if alpha == 0 else end).copy(), info
    # Endpoints must pass the rigid guard before any conversion to SO(3).
    # Record the small representation projection instead of silently repairing
    # invalid rotations. Original input arrays/files remain unchanged.
    r0, r1 = Rotation.from_matrix(start[:3, :3]), Rotation.from_matrix(end[:3, :3])
    info["rotation_roundoff_projection_max_abs"] = float(max(
        np.max(np.abs(r0.as_matrix() - start[:3, :3])),
        np.max(np.abs(r1.as_matrix() - end[:3, :3]))))
    output = np.eye(4)
    output[:3, :3] = (r0 * Rotation.from_rotvec(alpha * (r0.inv() * r1).as_rotvec())).as_matrix()
    output[:3, 3] = (1 - alpha) * start[:3, 3] + alpha * end[:3, 3]
    return rigid_matrix(output), info


def moving_rig_transform(target_from_source: np.ndarray, world_from_source_time: np.ndarray,
                         world_from_target_time: np.ndarray) -> np.ndarray:
    """Target at t2 <- source-camera frame at t2 <- world <- source at t1.

    Both temporal poses must describe the SAME camera in the SAME reference
    world. Do not substitute independently zeroed RGB and thermal odometry.
    """
    fixed = rigid_matrix(target_from_source)
    source = rigid_matrix(world_from_source_time)
    target = rigid_matrix(world_from_target_time)
    return rigid_matrix(fixed @ np.linalg.inv(target) @ source)
