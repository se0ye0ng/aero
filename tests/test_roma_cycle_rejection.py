import numpy as np

from scripts.probe_roma_cycle_rejection import admission, summarize


def test_inverse_wrong_translation_can_pass_so_not_physical_gt():
    points = np.array([[10.0, 10.0], [20.0, 20.0]])
    _, accepted, cycle = admission(
        lambda p: p + 10, lambda p: p - 10, points, (100, 100), (100, 100)
    )
    assert accepted.all()
    np.testing.assert_array_equal(cycle, 0)


def test_reflection_is_rejected_even_when_inverse_exact():
    def flip(p):
        return np.c_[100 - p[:, 0], p[:, 1]]

    _, accepted, _ = admission(flip, flip, np.array([[10.0, 10.0]]), (100, 100), (100, 100))
    assert not accepted.any()


def test_rejection_does_not_raise_total_pck():
    result = summarize([0, 1, 20, np.nan], [True, False, False, False])
    assert result["conditional_retained_pck3"] == 1
    assert result["retained_fraction"] == 0.25
    assert result["retained_pck3_all"] == 0.25
    assert result["original_pck3_all"] == 0.5
    assert result["rejected_good_points"] == 1
