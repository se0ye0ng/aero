from scripts.probe_roma_fresh_confirmation import aggregate, duplicate_labels


def test_duplicates_include_previous_and_within_new_panel():
    files = {}
    for pair, identity in (("old/1", "a"), ("new/5", "a"), ("new/6", "b"), ("other/5", "b")):
        for name in ("points_rgb.txt", "points_thermal.txt"):
            files[f"{pair}/{name}"] = identity + name
    assert duplicate_labels(["new/5", "new/6", "other/5"], ["old/1"], files) == ["new/5", "other/5"]


def test_empty_independent_subset_has_no_success_claim():
    summary = dict(
        landmarks=4, unprojectable=0, pck_all_landmarks={str(t): 1.0 for t in (1.0, 3.0, 5.0, 10.0)}
    )
    rows = [
        dict(
            pair="a",
            scores={
                k: [dict(summary=summary)]
                for k in ("fixed_confidence_ge_half", "ungated_diagnostic")
            },
        )
    ]
    result = aggregate(rows, ["a"])
    assert result["all_eligible"]["pairs"] == 1
    assert result["without_duplicate_labels"]["pairs"] == 0
    assert (
        result["without_duplicate_labels"]["variants"]["ungated_diagnostic"]["macro_pck"]["3.0"]
        is None
    )
