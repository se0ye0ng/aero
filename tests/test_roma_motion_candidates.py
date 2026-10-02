import numpy as np

from scripts.probe_roma_motion_candidates import candidates, select_pair


def test_bidirectional_ranking_changes_motion_choice_without_gt():
    a = np.zeros((20, 20), dtype=bool)
    a[2:4, 2:4] = True
    b = np.zeros_like(a)
    b[10:12, 10:12] = True
    selected, choices = select_pair([[a, b], [b, a]], [lambda p: p, lambda p: p])
    assert selected["indices"] == [0, 1]
    assert selected["score"] == 1
    assert len(choices) == 4


def test_abstain_on_no_reciprocal_support():
    mask = np.ones((4, 4), dtype=bool)
    selected, _ = select_pair(
        [[mask], [mask]], [lambda p: p, lambda p: np.full_like(p, np.nan, dtype=float)]
    )
    assert selected is None


def test_top_three_by_motion_score_only():
    mask = np.zeros((30, 30), dtype=bool)
    score = np.zeros_like(mask, dtype=float)
    for i in range(4):
        mask[i * 6 : i * 6 + 2, :2] = True
        score[i * 6 : i * 6 + 2, :2] = i + 1
    result = candidates(score, mask)
    assert len(result) == 3
    assert result[0][18:20, :2].all()
