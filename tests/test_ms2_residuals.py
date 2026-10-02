import numpy as np
import pytest

from scripts.analyze_ms2_residuals import (
    aggregate,
    bounded_disparity_error,
    residual_components,
    strata,
    summarize_mask,
)


def setup_geometry():
    k = np.diag([100.0, 100.0, 1.0])
    transform = np.eye(4)
    transform[0, 3] = -1.0
    return k, transform


def test_epipolar_decomposition_has_native_units_and_pythagorean_identity():
    k, transform = setup_geometry()
    source = np.array([[20.0, 30.0]])
    projected = np.array([[10.0, 30.0]])
    target = np.array([[14.0, 33.0]])
    cross, along = residual_components(source, target, projected, k, k, transform)
    np.testing.assert_allclose(abs(cross), [3.0])
    np.testing.assert_allclose(abs(along), [4.0])
    np.testing.assert_allclose(cross**2 + along**2, [25.0])


def test_depth_probe_cannot_repair_cross_line_error():
    k, transform = setup_geometry()
    source = np.array([[20.0, 30.0], [20.0, 30.0]])
    target = np.array([[9.5, 30.0], [9.5, 34.0]])
    result = bounded_disparity_error(source, target, np.array([10.0, 10.0]), 100.0, k, k, transform)
    np.testing.assert_allclose(result, [0.0, 4.0])


def test_depth_probe_clamps_to_interval_and_handles_infinity_endpoint():
    k, transform = setup_geometry()
    source = np.array([[20.0, 30.0], [20.0, 30.0]])
    result = bounded_disparity_error(
        source, np.array([[0.0, 30.0], [20.0, 30.0]]), np.array([10.0, 0.5]), 100.0, k, k, transform
    )
    np.testing.assert_allclose(result, [9.0, 0.0])


def test_radius_zero_is_original_projection_and_pure_rotation_segment_degenerate():
    k, transform = setup_geometry()
    source = np.array([[20.0, 30.0]])
    result = bounded_disparity_error(
        source, np.array([[14.0, 33.0]]), np.array([10.0]), 100.0, k, k, transform, radius=0
    )
    np.testing.assert_allclose(result, [5.0])
    result = bounded_disparity_error(
        source, np.array([[24.0, 33.0]]), np.array([10.0]), 100.0, k, k, np.eye(4)
    )
    np.testing.assert_allclose(result, [5.0])


def test_invalid_intervals_and_line_geometry_not_silently_counted_as_pass():
    k, transform = setup_geometry()
    source = np.array([[20.0, 30.0]])
    invalid = bounded_disparity_error(source, source, np.array([np.nan]), 100.0, k, k, transform)
    assert np.isinf(invalid).all()
    cross, along = residual_components(source, source, source, k, k, np.eye(4))
    assert np.isnan(cross).all() and np.isnan(along).all()
    stats = summarize_mask(np.array([True]), invalid, cross, along, invalid)
    assert stats["points"] == stats["failed_pck3"] == stats["invalid_projection"] == 1
    assert stats["pck3"] == stats["optimistic_disparity_radius1_pck3"] == 0
    assert stats["invalid_line_geometry"] == stats["invalid_disparity_interval"] == 1


def test_bin_partitions_preserve_population_at_boundaries():
    source = np.array([[0.0, 0.0], [408.0, 192.0], [816.0, 383.0], [1223.0, 191.0]])
    masks = strata(np.array([1.0, 10.0, 20.0, 40.0]), np.array([0.0, 0.25, 0.5, 0.75]), source)
    for prefix in ("source_cell_", "depth_m_", "confidence_"):
        np.testing.assert_array_equal(
            sum(v.astype(int) for k, v in masks.items() if k.startswith(prefix)), np.ones(4)
        )


def test_invalid_radius_rejected():
    k, transform = setup_geometry()
    with pytest.raises(ValueError):
        bounded_disparity_error(
            np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 100.0, k, k, transform, radius=-1
        )


def test_empty_stratum_retains_frame_count_and_does_not_invent_score():
    empty = summarize_mask(np.zeros(0, bool), *(np.zeros(0) for _ in range(4)))
    row = dict(model="model", condition="paired", candidate="candidate", strata={"empty": empty})
    result = aggregate([row])[0]
    assert result["planned_frames"] == 1 and result["represented_frames"] == 0
    assert result["points"] == 0
    assert result["equal_represented_frame_scores"]["pck3"] is None
