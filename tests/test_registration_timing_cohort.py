import copy

import pytest

from scripts.probe_registration_timing_cohort import stable_hypothesis, summarize


def result():
    return {
        "status": "screened_not_calibrated",
        "blocks": [{"selected_lag": 2}] * 4,
        "all_four_selected_same_lag": True,
        "all_four_check_improve_at_least_10percent": True,
        "any_selected_search_boundary": False,
    }


@pytest.mark.parametrize("defect", ["zero", "boundary", "inconsistent", "no_gain", "missing"])
def test_nonzero_consistent_nonboundary_supported_hypothesis_required(defect):
    value = copy.deepcopy(result())
    assert stable_hypothesis(value)
    if defect == "zero":
        value["blocks"] = [{"selected_lag": 0}] * 4
    elif defect == "boundary":
        value["any_selected_search_boundary"] = True
    elif defect == "inconsistent":
        value["all_four_selected_same_lag"] = False
    elif defect == "no_gain":
        value["all_four_check_improve_at_least_10percent"] = False
    else:
        value = {"status": "insufficient_common_frames"}
    assert not stable_hypothesis(value)


def test_summary_keeps_unobservable_sequences_and_negative_controls():
    absent = {"status": "unobservable_degenerate_trajectory"}
    rows = [
        {
            "sequence_id": "a",
            "v7_geometry_joint_pass": False,
            "actual": result(),
            "shuffled": absent,
        },
        {
            "sequence_id": "b",
            "v7_geometry_joint_pass": True,
            "actual": absent,
            "shuffled": result(),
        },
    ]
    summary = summarize(rows)
    assert summary["actual"]["v7_fail"]["stable_lags"] == {"a": 2}
    assert summary["actual"]["v7_pass"]["sequences"] == 1
    assert summary["actual"]["v7_pass"]["stable_lag_hypotheses"] == 0
    assert summary["shuffled"]["v7_pass"]["stable_sequence_ids"] == ["b"]
