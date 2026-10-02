import pytest

from scripts import probe_registration_fit_capacity as probe


def row(sid, failures):
    return {"sequence_id": sid, **{d: failures for d in probe.DIRECTIONS}}


def test_panel_excludes_other_failures_and_uses_distinct_prefixes(monkeypatch):
    monkeypatch.setattr(probe, "failures", set)
    rows = [
        row("a_2", ["bbox_iou"]),
        row("a_1", ["absolute_area_ratio_change"]),
        row("b_1", []),
        row("b_2", ["bbox_iou", "valid_fraction"]),
        row("c_1", ["centroid_shift_fraction"]),
        row("d_1", ["bbox_iou"]),
    ]
    selected = probe.alignment_failure_panel(rows, count=2)
    assert [r["sequence_id"] for r in selected] == ["a_1", "c_1"]
    assert selected == probe.alignment_failure_panel(rows[::-1], count=2)
    with pytest.raises(ValueError, match="not enough"):
        probe.alignment_failure_panel(rows, count=4)


def test_panel_rejects_duplicates_and_invalid_size(monkeypatch):
    monkeypatch.setattr(probe, "failures", set)
    item = row("a_1", ["bbox_iou"])
    with pytest.raises(ValueError, match="duplicate"):
        probe.alignment_failure_panel([item, item], count=1)
    with pytest.raises(ValueError, match="positive"):
        probe.alignment_failure_panel([item], count=0)
