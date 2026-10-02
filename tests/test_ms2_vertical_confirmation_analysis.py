import itertools

import pytest

from scripts.analyze_ms2_vertical_confirmation import (
    CONDITIONS,
    OFFSETS,
    common_effects,
    require_complete_cases,
)


def cases():
    return [
        dict(sensor=s, frame=f, condition=c, offset=o)
        for s, f, c, o in itertools.product(("rgb", "thr"), ["000174"], CONDITIONS, OFFSETS)
    ]


def test_complete_case_inventory():
    require_complete_cases(cases(), ["000174"])


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unplanned"])
def test_bad_case_inventory(mode):
    rows = cases()
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows[-1] = rows[0].copy()
    else:
        rows[-1]["offset"] = 1.0
    with pytest.raises(ValueError):
        require_complete_cases(rows, ["000174"])


def test_common_effects_retains_unavailable_frames():
    rows = [
        dict(
            sensor="thr",
            condition="paired_timed",
            all_reference_points=10,
            common_points=5,
            scores={
                str(o): {"conditional_median_abs_px": v}
                for o, v in zip(OFFSETS, [2.0, 1.0, 3.0], strict=True)
            },
        ),
        dict(
            sensor="thr",
            condition="paired_timed",
            all_reference_points=20,
            common_points=0,
            scores={},
        ),
    ]
    result = next(
        r
        for r in common_effects(rows)
        if r["sensor"] == "thr" and r["condition"] == "paired_timed" and r["offset"] == 0.5
    )
    assert result["planned_frames"] == 2
    assert result["reference_points"] == 30
    assert result["frames_with_common_points"] == 1
    assert result["improved_frame_count"] == 1
    assert result["median_of_common_frame_medians_px"] == 1.0
