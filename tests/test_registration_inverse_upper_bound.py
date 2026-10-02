from copy import deepcopy

from aero_ir.registration.qualification_v4 import THRESHOLDS
from scripts.registration_inverse_upper_bound import bound


def test_cycle_repair_cannot_raise_fixed_box_bound():
    good = {
        "box_in_bounds": True,
        "bbox_iou": 0.8,
        "centroid_shift_fraction": 0.05,
        "absolute_area_ratio_change": 0.1,
        "valid_fraction": 1.0,
        "positive_jacobian_fraction": 1.0,
        "roi_positive_jacobian_fraction": 1.0,
        "cycle_p95_pixels": 20.0,
    }
    bad = {**good, "bbox_iou": 0.4}
    rows = [{"sequence_id": "one", "forward": good}, {"sequence_id": "two", "forward": bad}]
    before = bound(rows, "forward", THRESHOLDS)
    repaired = deepcopy(rows)
    for row in repaired:
        row["forward"]["cycle_p95_pixels"] = 0.0
    assert before == bound(repaired, "forward", THRESHOLDS)
    assert before["joint_frame_pass_upper_bound"] == 0.5
    assert not before["can_reach_frozen_threshold"]


def test_missing_roi_jacobian_is_a_fixed_field_failure():
    row = {
        "box_in_bounds": True,
        "bbox_iou": 0.8,
        "centroid_shift_fraction": 0.05,
        "absolute_area_ratio_change": 0.1,
        "valid_fraction": 1.0,
        "positive_jacobian_fraction": 1.0,
        "roi_positive_jacobian_fraction": None,
    }
    result = bound([{"sequence_id": "one", "forward": row}], "forward", THRESHOLDS)
    assert result["joint_frame_pass_upper_bound"] == 0.0
