import numpy as np

from scripts.analyze_ms2_image_matching import direct_reference_scores


def reference():
    return dict(
        source_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
        target_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
        geometric_mask=np.array([True, True]),
        zbuffer_mask=np.array([True, True]),
        depth_consistency_mask=np.array([False, True]),
    )


def test_nearest_anchor_not_reselected_after_mask():
    result = direct_reference_scores(np.array([[0.1, 0.0]]), np.array([[0.0, 0.0]]), reference())
    assert result["slices"]["geometric_mask"]["0.25"]["associated_matches"] == 1
    assert result["slices"]["depth_consistency_mask"]["1.0"]["associated_matches"] == 0


def test_large_target_error_is_retained():
    result = direct_reference_scores(np.array([[0.0, 0.0]]), np.array([[100.0, 0.0]]), reference())
    selected = result["slices"]["geometric_mask"]["0.25"]
    assert selected["associated_matches"] == 1
    assert selected["within_target_native_px_counts"]["10.0"] == 0


def test_empty_matches():
    result = direct_reference_scores(np.empty((0, 2)), np.empty((0, 2)), reference())
    assert result == {"image_matches": 0, "slices": {}}
