import types

import numpy as np
import pytest
import torch

from aero_ir.registration.detector_free import (
    guard_empty_xoftr_fine,
    header_diagnostic,
    homography_diagnostic,
    inside_box,
    pair_metrics,
    reciprocal_mask,
)


def test_roi_routing_is_separate_from_reciprocity():
    a = np.array([[1.0, 1.0], [3.0, 3.0], [8.0, 8.0]])
    b = np.array([[2.0, 2.0], [9.0, 9.0], [10.0, 10.0]])
    box = np.array([0, 0, 5, 5])
    stats, reciprocal = pair_metrics(a, b, b, a, box, box)
    assert reciprocal.all()
    assert stats["source_box_matches"] == 2
    assert stats["both_box_matches"] == 1
    assert stats["one_box_only_matches"] == 1
    assert stats["box_routing_fraction_not_precision"] == 0.5


def test_empty_matches_are_not_perfect_consistency():
    empty = np.empty((0, 2))
    box = np.array([0, 0, 5, 5])
    stats, reciprocal = pair_metrics(empty, empty, empty, empty, box, box)
    assert len(reciprocal) == 0
    assert stats["reciprocal_fraction"] is None
    assert stats["box_routing_fraction_not_precision"] is None
    assert stats["homography_consistency_not_accuracy"]["status"] == "insufficient_matches"


def test_reciprocal_checks_both_endpoints():
    a, b = np.array([[2.0, 4.0]]), np.array([[3.0, 5.0]])
    assert reciprocal_mask(a, b, b + 1, a + 1).all()
    assert not reciprocal_mask(a, b, b + 1, a + 3).any()


def test_homography_checks_held_out_matches():
    source = np.random.default_rng(5).uniform(0, 100, (40, 2))
    target = source * 1.1 + np.array([10, -2])
    result = homography_diagnostic(source, target)
    assert result["fit_count"] == result["check_count"] == 20
    assert result["check_fraction_within_3px"] == 1
    assert result["check_median_reprojection_px"] < 1e-4


def test_xoftr_placeholder_suppressed_without_changing_valid_branch():
    calls = []
    fine = types.SimpleNamespace(
        fine_thr=0.1,
        get_fine_sub_match=lambda *args: calls.append(True) or {"valid": True},
    )
    guard_empty_xoftr_fine(types.SimpleNamespace(fine_matching=fine))
    result = fine.get_fine_sub_match(torch.zeros(1, 25, 25), None, None, {})
    assert fine.suppressed_placeholder and len(result["mkpts0_f"]) == 0
    assert not calls
    result = fine.get_fine_sub_match(torch.ones(1, 25, 25), None, None, {})
    assert result == {"valid": True} and len(calls) == 1


@pytest.mark.parametrize("points, expected", [([[2, 3]], [True]), ([[20, 3]], [False])])
def test_box_coordinate_units(points, expected):
    assert inside_box(np.array(points), np.array([0, 0, 10, 10])).tolist() == expected


def test_header_screen_excludes_either_endpoint_but_does_not_claim_accuracy():
    p0 = np.array([[50.0, 10.0], [20.0, 50.0], [25.0, 55.0]])
    p1 = np.array([[50.0, 50.0], [20.0, 5.0], [25.0, 60.0]])
    result = header_diagnostic(p0, p1, 100, 100)
    assert result["header_either_matches"] == 2
    assert result["non_header_matches"] == 1


def test_spatial_audit_rejects_homography_poles_and_collinear_support():
    from scripts.analyze_detector_free_screen import hull_fraction, transfer_iou

    box = np.array([0.0, 0.0, 10.0, 10.0])
    assert transfer_iou(np.eye(3), box, box) == pytest.approx(1)
    assert hull_fraction(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), box) == 0
    pole = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -5.0]])
    assert transfer_iou(pole, box, box) is None
