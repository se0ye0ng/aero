import numpy as np

from scripts.analyze_roma_geometry import audit_queries, box_grid


def test_inverse_can_be_wrong_but_perfectly_consistent():
    q = box_grid([10, 10, 20, 20])
    r = audit_queries(lambda p: p + [100, 0], lambda p: p - [100, 0], q, [10, 10, 20, 20])
    assert r["cycle_le_3_all_fraction"] == 1
    assert r["target_box_hits"] == 0
    assert r["cycle_le_3_but_outside_target_box"] == 81
    assert r["jacobian_nonpositive"] == 0


def test_reflection_detected_despite_zero_cycle():
    q = box_grid([10, 10, 20, 20])

    def flip(p):
        return np.c_[30 - p[:, 0], p[:, 1]]

    r = audit_queries(flip, flip, q, [10, 10, 20, 20])
    assert r["cycle_le_3_all_fraction"] == 1
    assert r["jacobian_nonpositive"] == 81


def test_unsupported_queries_not_successes():
    r = audit_queries(
        lambda p: np.full_like(p, np.nan), lambda p: p, box_grid([0, 0, 10, 10]), [0, 0, 10, 10]
    )
    assert r["cycle_supported"] == 0
    assert r["cycle_le_3_all_fraction"] == 0
    assert r["jacobian_nonpositive_supported_fraction"] is None
