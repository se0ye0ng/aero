import numpy as np
import pytest

from aero_ir.registration.ms2_image_reference import (
    display_thermal,
    score_map,
    sparse_reference,
    thermal_window,
)


def test_window_is_shared_not_refit_per_frame():
    images = [
        np.arange(100, dtype=np.uint16).reshape(10, 10),
        np.arange(100, 200, dtype=np.uint16).reshape(10, 10),
    ]
    window = thermal_window(images)
    a, b = [display_thermal(im, window) for im in images]
    assert a.max() < b.max()
    assert a.min() < b.min()
    assert images[0][0, 1] == 1  # Raw counts unchanged.


def test_constant_window_rejected():
    with pytest.raises(ValueError):
        thermal_window([np.ones((3, 3), dtype=np.uint16)])


def test_identity_and_missing_prediction_denominator():
    depth = np.full((3, 3), 256, dtype=np.uint16)
    ref = sparse_reference(depth, depth, np.eye(3), np.eye(3), np.eye(4))
    pred = ref["source_xy"].copy()
    pred[0] = np.nan
    result = score_map(pred, ref)
    for item in result["masks"].values():
        assert item["references"] == 9
        assert item["unsupported_predictions"] == 1
        assert item["fraction_within_target_native_px"]["1.0"] == 8 / 9


def test_inconsistent_depth_does_not_disappear_from_broad_reference():
    depth = np.full((3, 3), 256, dtype=np.uint16)
    ref = sparse_reference(depth, depth * 2, np.eye(3), np.eye(3), np.eye(4))
    result = score_map(ref["source_xy"], ref)
    assert result["masks"]["geometric_mask"]["references"] == 9
    assert result["masks"]["depth_consistency_mask"]["references"] == 0


def test_empty_depth_has_no_fabricated_reference():
    depth = np.zeros((3, 3), dtype=np.uint16)
    ref = sparse_reference(depth, depth, np.eye(3), np.eye(3), np.eye(4))
    result = score_map(np.empty((0, 2)), ref)
    assert result["all_source_depth_points"] == 0
    assert result["masks"]["geometric_mask"]["fraction_within_target_native_px"]["3.0"] is None
