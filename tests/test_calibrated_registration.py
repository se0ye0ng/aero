import numpy as np
import pytest

from aero_ir.registration.calibrated import (
    cropped_resized_intrinsics,
    from_millimetres,
    intrinsic_matrix,
    ms2_depth_metres,
    project_pixels,
    rigid_matrix,
    via_common_camera,
    zbuffer_candidates,
)


def k(f=100.0, cx=10.0, cy=10.0):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def project(points, depths, transform=None, source_k=None, target_k=None, shape=(32, 32)):
    return project_pixels(
        np.asarray(points),
        np.asarray(depths),
        k() if source_k is None else source_k,
        k() if target_k is None else target_k,
        np.eye(4) if transform is None else transform,
        source_shape=shape,
        target_shape=shape,
    )


def test_identity_has_no_half_pixel_shift_and_preserves_depth():
    points = np.array([[0.0, 0.0], [10.25, 12.75], [31.0, 31.0]])
    depths = np.array([1.0, 2.0, 3.0])
    result = project(points, depths)
    np.testing.assert_allclose(result.target_xy, points, atol=1e-12)
    np.testing.assert_allclose(result.target_z_m, depths)
    assert result.supported.all()


def test_identity_boundary_roundoff_does_not_discard_valid_pixels():
    rng = np.random.default_rng(0)
    points = np.vstack(
        [
            np.column_stack([np.zeros(100), rng.uniform(0, 255, 100)]),
            np.column_stack([np.full(100, 639.0), rng.uniform(0, 255, 100)]),
        ]
    )
    matrix = np.array([[723.479, 0, 312.298], [0, 698.173, 121.585], [0, 0, 1.0]])
    result = project(
        points,
        rng.uniform(0.1, 200, len(points)),
        source_k=matrix,
        target_k=matrix,
        shape=(256, 640),
    )
    np.testing.assert_allclose(result.target_xy, points, atol=1e-10)
    assert result.supported.all()
    assert zbuffer_candidates(result, (256, 640)).sum() > 0
    outside = project(
        [[-1e-6, 5], [639 + 1e-6, 5]],
        [1.0, 1.0],
        source_k=matrix,
        target_k=matrix,
        shape=(256, 640),
    )
    assert not outside.supported.any()


def test_ms2_units_and_common_camera_direction():
    rgb_from_nir = from_millimetres(np.eye(3), [100.0, 0.0, 0.0])
    ir_from_nir = from_millimetres(np.eye(3), np.array([[200.0], [0.0], [0.0]]))
    rgb_from_ir = via_common_camera(rgb_from_nir, ir_from_nir)
    np.testing.assert_allclose(rgb_from_ir[:3, 3], [-0.1, 0, 0])
    result = project([[10.0, 10.0]], [10.0], transform=rgb_from_ir)
    np.testing.assert_allclose(result.target_xy, [[9.0, 10.0]])


def test_metric_axis_depth_not_range_and_not_single_homography():
    transform = from_millimetres(np.eye(3), [100.0, 0.0, 0.0])
    result = project([[10.0, 10.0], [10.0, 10.0]], [1.0, 2.0], transform=transform)
    np.testing.assert_allclose(result.target_xy, [[20.0, 10.0], [15.0, 10.0]])
    off_axis = project([[15.0, 15.0]], [2.0], transform=transform)
    np.testing.assert_allclose(off_axis.target_xy, [[20.0, 15.0]])


def test_round_trip_control_uses_target_z_and_full_inverse():
    angle = 0.1
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    transform = from_millimetres(rotation, [50, -30, 100])
    points, depth = np.array([[11.0, 12.0], [15.0, 16.0]]), np.array([2.0, 3.0])
    forward = project(points, depth, transform=transform)
    reverse = project(forward.target_xy, forward.target_z_m, transform=np.linalg.inv(transform))
    assert forward.supported.all() and reverse.supported.all()
    np.testing.assert_allclose(reverse.target_xy, points, atol=1e-12)
    np.testing.assert_allclose(reverse.target_z_m, depth, atol=1e-12)


def test_common_camera_composition_order_with_rotation():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    source = from_millimetres(rotation, [100.0, 200.0, 0.0])
    target = from_millimetres(np.eye(3), [300.0, 400.0, 0.0])
    point_common = np.array([0.1, 0.2, 3.0, 1.0])
    actual = via_common_camera(target, source) @ (source @ point_common)
    np.testing.assert_allclose(actual, target @ point_common)


def test_crop_then_resize_intrinsics_use_pixel_centres():
    result = cropped_resized_intrinsics(k(cx=20, cy=15), (2.0, 3.0), (0.5, 0.25))
    np.testing.assert_allclose(result, [[50.0, 0, 8.75], [0, 25.0, 2.625], [0, 0, 1.0]])


def test_crop_resize_commutes_with_projection_for_supported_points():
    points = np.array([[10.0, 12.0], [22.0, 24.0]])
    depth = np.array([2.0, 3.0])
    transform = from_millimetres(np.eye(3), [10.0, 20.0, 50.0])
    native = project(points, depth, transform=transform)
    source_crop, target_crop = np.array([2.0, 3.0]), np.array([1.0, 2.0])
    source_scale, target_scale = np.array([0.5, 0.5]), np.array([0.75, 0.5])
    resized = project_pixels(
        (points - source_crop + 0.5) * source_scale - 0.5,
        depth,
        cropped_resized_intrinsics(k(), source_crop, source_scale),
        cropped_resized_intrinsics(k(), target_crop, target_scale),
        transform,
        source_shape=(14, 15),
        target_shape=(15, 23),
    )
    assert resized.supported.all()
    np.testing.assert_allclose(
        resized.target_xy, (native.target_xy - target_crop + 0.5) * target_scale - 0.5
    )


def test_depth_encoding_preserves_holes_without_temperature_conversion():
    encoded = np.array([[0, 256, 512, 65535]], dtype=np.uint16)
    decoded = ms2_depth_metres(encoded)
    assert np.isnan(decoded[0, 0])
    np.testing.assert_allclose(decoded[0, 1:], [1.0, 2.0, 65535 / 256.0])
    np.testing.assert_array_equal(encoded, [[0, 256, 512, 65535]])
    with pytest.raises(ValueError, match="uint16"):
        ms2_depth_metres(encoded.astype(np.uint8))


def test_invalid_depth_and_unsupported_pixels_cannot_become_identity():
    result = project(
        [[10.0, 10.0]] * 4 + [[-1, 10], [32, 10], [np.nan, 10]], [0, -1, np.nan, np.inf, 1, 1, 1]
    )
    assert not result.supported.any()
    assert np.isnan(result.target_xy).all()
    assert not zbuffer_candidates(result, (32, 32)).any()


def test_behind_camera_and_out_of_frame_are_separate_failures():
    behind = project([[10.0, 10.0]], [1.0], transform=from_millimetres(np.eye(3), [0, 0, -2000]))
    assert behind.source_valid[0] and not behind.target_in_front[0]
    assert not behind.supported[0]
    outside = project([[10.0, 10.0]], [1.0], transform=from_millimetres(np.eye(3), [1000, 0, 0]))
    assert outside.source_valid[0] and outside.target_in_front[0]
    assert not outside.target_in_bounds[0] and not outside.supported[0]


def test_zbuffer_rejects_farther_sample_without_claiming_unobserved_visibility():
    result = project(
        [[1.0, 1.0], [1.5, 1.0]],
        [1.0, 2.0],
        source_k=np.eye(3),
        target_k=np.eye(3),
        transform=from_millimetres(np.eye(3), [1000, 0, 0]),
    )
    np.testing.assert_allclose(result.target_xy, [[2, 1], [2, 1]])
    np.testing.assert_array_equal(zbuffer_candidates(result, (32, 32)), [True, False])
    assert not zbuffer_candidates(result, (1, 1)).any()
    with pytest.raises(ValueError, match="tolerance"):
        zbuffer_candidates(result, (32, 32), tie_tolerance_m=-1)


def test_empty_input_remains_empty():
    result = project(np.empty((0, 2)), np.empty(0))
    assert result.target_xy.shape == (0, 2)
    assert zbuffer_candidates(result, (32, 32)).shape == (0,)


def test_bad_camera_matrices_fail_without_silent_repair():
    for transform in (np.diag([0.6, 1, 1, 1]), np.diag([-1.0, 1, 1, 1])):
        with pytest.raises(ValueError, match="orthogonal"):
            rigid_matrix(transform)
    for matrix in (np.diag([-1.0, 1, 1]), np.zeros((3, 3)), np.ones((2, 2))):
        with pytest.raises(ValueError, match="pinhole"):
            intrinsic_matrix(matrix)
    with pytest.raises(ValueError, match="scales"):
        cropped_resized_intrinsics(k(), (0, 0), (0, 1))
    with pytest.raises(ValueError, match="Nx2"):
        project([[1.0, 1.0]], [])
    with pytest.raises(ValueError, match="shape"):
        project([[1.0, 1.0]], [1.0], shape=(0, 0))
