import pytest

from scripts import screen_registration_scale_learning as screen


def test_paired_passes_gains_losses_and_identity_validation(monkeypatch):
    monkeypatch.setattr(screen, "joint", lambda row: row["passed"])
    initial = [{"sequence_id": str(i), "passed": p} for i, p in enumerate([True, False, True])]
    candidate = [{"sequence_id": str(i), "passed": p} for i, p in enumerate([False, True, True])]
    result = screen.paired_summary(initial, candidate)
    assert result["joint_passes"] == 2
    assert result["joint_pass_rate"] == pytest.approx(2 / 3)
    assert result["gained_ids"] == ["1"]
    assert result["lost_ids"] == ["0"]
    with pytest.raises(ValueError, match="unmatched"):
        screen.paired_summary(initial, candidate[::-1])


def test_verifier_rejects_non_diagnostic_before_loading_files(tmp_path):
    with pytest.raises(ValueError, match="not a scale-learning"):
        screen.verified_heads(tmp_path, {"kind": "qualified"})
    with pytest.raises(ValueError, match="budget"):
        screen.verified_heads(
            tmp_path,
            {
                "kind": "matched_scale_consistency_cpu_learning_diagnostic",
                "settings": {"updates_per_arm": 1},
            },
        )
