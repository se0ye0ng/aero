import numpy as np
import pytest

from scripts.analyze_registration_target_support import attribute


def test_target_support_attribution_does_not_treat_box_membership_as_accuracy():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])
    info = dict(status="fit", control_match_indices=[0, 1, 2, 3])
    result = attribute(points, points + 1, info, [3, 3, 7, 7], [3, 3, 7, 7], 0.1)
    assert result["both_box_matches"] == 1
    assert result["controls_in_source_box"] == 0
    assert result["source_corners_inside_control_hull"] == 4
    assert result["failure_stage"] == "box_proxy_iou_low"


def test_missing_hull_support_is_not_silently_scored():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    info = dict(status="fit", control_match_indices=[0, 1, 2, 3])
    args = (points, points, info, [-1, -1, 11, 11], [1, 1, 2, 2])
    assert attribute(*args, None)["failure_stage"] == "source_corner_outside_control_hull"
    with pytest.raises(ValueError, match="contradicts"):
        attribute(*args, 0.8)


def test_missing_fit_and_invalid_mapped_corners_remain_failures():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    info = dict(status="insufficient_matches")
    assert (
        attribute(points, points, info, [1, 1, 2, 2], [1, 1, 2, 2], None)["failure_stage"]
        == "no_warp_fit"
    )
    info = dict(status="fit", control_match_indices=[0, 1, 2, 3])
    assert (
        attribute(points, points, info, [1, 1, 2, 2], [1, 1, 2, 2], None)["failure_stage"]
        == "mapped_corner_unavailable"
    )
