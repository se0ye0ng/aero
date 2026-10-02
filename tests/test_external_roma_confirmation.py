from scripts.probe_external_roma_confirmation import aggregate


def test_duplicate_exclusion_does_not_change_primary_denominator():
    rows = []
    for name, pck in (("scene/2", 1.0), ("scene/3", 0.0)):
        summary = dict(
            landmarks=4,
            unprojectable=0,
            pck_all_landmarks={str(t): pck for t in (1.0, 3.0, 5.0, 10.0)},
        )
        rows.append(
            dict(
                pair=name,
                scores={
                    k: [dict(summary=summary)]
                    for k in ("fixed_confidence_ge_half", "ungated_diagnostic")
                },
            )
        )
    result = aggregate(rows, ["scene/2"])
    assert result["all12"]["variants"]["ungated_diagnostic"]["macro_pck"]["3.0"] == 0.5
    assert result["without_development_duplicate_labels"]["pairs"] == 1
    assert (
        result["without_development_duplicate_labels"]["variants"]["ungated_diagnostic"][
            "macro_pck"
        ]["3.0"]
        == 0
    )
