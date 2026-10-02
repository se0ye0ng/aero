import numpy as np

from scripts.probe_xoftr_overlay import aggregate, header_rows, restore, retained


def test_exact_crop_and_coordinate_restore():
    assert header_rows(360) == 72
    assert header_rows(512) == 104
    p = np.array([[0.0, 0.0], [639.0, 407.0]])
    assert np.array_equal(restore(p, 104), [[0.0, 104.0], [639.0, 511.0]])
    assert np.array_equal(restore(p, 104) - [0, 104], p)


def test_common_domain_is_both_endpoints_and_empty_safe():
    a = np.array([[1.0, 72.0], [1.0, 71.0], [1.0, 73.0]])
    b = np.array([[1.0, 104.0], [1.0, 105.0], [1.0, 103.0]])
    assert retained(a, b, [72, 104]).tolist() == [True, False, False]
    assert retained(a[:0], b[:0], [72, 104]).shape == (0,)
    assert aggregate([]) == {}
