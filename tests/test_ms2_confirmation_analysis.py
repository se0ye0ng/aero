from copy import deepcopy

import pytest

from scripts.analyze_ms2_confirmation import aggregate, compare_values


def rows_for_frame():
    rows = []
    for model in ("xoftr_640", "minima_xoftr"):
        for condition in ("paired", "unrelated_thermal"):
            stats = dict(
                points=4, supported=4, fraction_all={"3.0": 0.5}, conditional_median_px=2.0
            )
            scores = {
                c: dict(all_matches=deepcopy(stats), fixed_supported=deepcopy(stats))
                for c in ("author_calibration", "candidate")
            }
            scores["candidate"]["fixed_supported"].update(
                supported=0, fraction_all={"3.0": 0.0}, conditional_median_px=None
            )
            rows.append(dict(model=model, frame="000001", condition=condition, scores=scores))
    return rows


def test_failed_candidates_are_not_removed_from_aggregate():
    summary = aggregate(rows_for_frame(), ["000001"], ["author_calibration", "candidate"])
    item = next(x for x in summary if x["candidate"] == "candidate")
    assert item["fixed_reference_points"] == 4
    assert item["candidate_projection_failures"] == 4
    assert item["equal_frame_score_missing_as_zero"] == 0
    assert item["worsened_frames"] == 1


def test_missing_case_and_changed_denominator_rejected():
    rows = rows_for_frame()
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(rows[:-1], ["000001"], ["author_calibration", "candidate"])
    rows[0]["scores"]["candidate"]["fixed_supported"]["points"] = 0
    with pytest.raises(ValueError, match="denominators"):
        aggregate(rows, ["000001"], ["author_calibration", "candidate"])


def test_saved_statistics_must_equal_reconstruction():
    compare_values({"median": None, "score": 0.5}, {"median": None, "score": 0.5})
    with pytest.raises(ValueError):
        compare_values({"score": 0.51}, {"score": 0.5})
