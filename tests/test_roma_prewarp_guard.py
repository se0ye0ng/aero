import numpy as np

from scripts.probe_roma_prewarp_guard import guarded


def test_rejected_motion_preserves_coarse_query():
    q = np.array([[10.0, 10.0]])
    predict = guarded(lambda p: p + 10, lambda p: p - 10, (100, 100), 3)
    np.testing.assert_array_equal(predict(q), q)


def test_inconsistent_small_motion_is_rejected():
    q = np.array([[10.0, 10.0]])
    predict = guarded(lambda p: p + 2, lambda p: p, (100, 100), 3)
    np.testing.assert_array_equal(predict(q), q)


def test_small_inverse_consistent_motion_can_pass_without_physical_correctness():
    q = np.array([[10.0, 10.0]])
    predict = guarded(lambda p: p + 1, lambda p: p - 1, (100, 100), 3)
    np.testing.assert_allclose(predict(q), q + 1)
