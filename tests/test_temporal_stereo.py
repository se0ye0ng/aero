import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aero_ir.registration.temporal_stereo import (
    estimate_pair,
    native_to_rectified,
    points_from_disparity,
    rectification,
    sample_native_depth,
)


def geometry(rotated=False):
    k = np.array([[150.0, 0.0, 160.0], [0.0, 150.0, 48.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[0, 3] = -0.3
    if rotated:
        transform[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.03]).as_matrix()
        transform[1:, 3] = [0.02, 0.01, 1.0]
    return rectification(k, k, transform, (96, 320))


@pytest.mark.parametrize("rotated", [False, True])
def test_3d_points_reconstruct_in_native_camera_not_virtual_rectified_camera(rotated):
    g = geometry(rotated)
    original = np.array([[-0.5, 0.1, 5.0], [1.0, -0.2, 10.0], [0.8, 0.4, 8.0]])
    native_h = original @ g["left_k"].T
    native = native_h[:, :2] / native_h[:, 2, None]
    left = native_to_rectified(native, g)
    right_camera = original @ g["right_from_left"][:3, :3].T + g["right_from_left"][:3, 3]
    right_h = right_camera @ g["r2"].T @ g["p2"][:, :3].T
    right = right_h[:, :2] / right_h[:, 2, None]
    np.testing.assert_allclose(left[:, 1], right[:, 1], atol=1e-10)
    reconstructed = points_from_disparity(native, left[:, 0] - right[:, 0], g)
    np.testing.assert_allclose(reconstructed, original, atol=1e-10)


def test_rectified_known_disparity_depth_and_border_support():
    g = geometry()
    rng = np.random.default_rng(14)
    left = rng.integers(0, 256, (96, 320), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]
    stereo = estimate_pair(left, right, g)
    assert not stereo["valid"][:2].any()
    assert not stereo["valid"][-2:].any()
    # With128disparities on320pixels, both forward and reversed searches only
    # overlap in the central band. Unsupported search borders are not filled.
    yy, xx = np.mgrid[20:76:4, 150:190:4]
    depth, valid = sample_native_depth(stereo, g, np.c_[xx.ravel(), yy.ravel()])
    assert valid.mean() > 0.9
    assert abs(np.median(depth[valid]) - 150 * 0.3 / 8) < 0.02


def test_zero_or_negative_disparity_and_missing_support_not_depth():
    g = geometry()
    points = np.array([[170.0, 30.0], [180.0, 40.0]])
    assert np.isnan(points_from_disparity(points, [0.0, -1.0], g)).all()
    stereo = dict(disparity=np.ones((96, 320)), valid=np.zeros((96, 320), bool))
    depth, valid = sample_native_depth(stereo, g, points)
    assert not valid.any() and np.isnan(depth).all()


def test_degenerate_baseline_and_nonfinite_points_rejected():
    with pytest.raises(ValueError, match="baseline"):
        rectification(np.eye(3), np.eye(3), np.eye(4), (96, 320))
    with pytest.raises(ValueError, match="finite"):
        native_to_rectified([[np.nan, 1.0]], geometry())
