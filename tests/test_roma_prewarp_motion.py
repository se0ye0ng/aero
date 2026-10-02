import numpy as np
import pytest

from scripts.analyze_roma_prewarp_motion import summarize_motion


def test_large_inverse_consistent_motion_is_not_correctness():
    query = np.array([[10.0, 10.0], [20.0, 20.0]])
    result = summarize_motion(query, query + 100, query, 20, [True, True])
    assert result["cycle_le_one_and_motion_gt_box_diagonal"] == 2
    assert result["queries"] == 2


def test_invalid_observations_keep_denominator():
    q = np.array([[10.0, 10.0], [np.nan, 20.0]])
    result = summarize_motion(q, q, q, 20, [False, True])
    assert result["queries"] == 2
    assert result["observed_cycle_queries"] == 0
    assert result["median_motion_pixels"] is None
    with pytest.raises(ValueError):
        summarize_motion(q, q, q, 0, [True, True])
