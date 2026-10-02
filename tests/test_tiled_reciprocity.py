import numpy as np
import pytest

from scripts.probe_antiuav_tiled_reciprocity import reciprocal_ids


def test_swapped_reverse_coordinates_and_outlier():
    p = np.array([[1, 2], [10, 20], [40, 50]])
    q = p + [8, -8]
    ids, reverse = reciprocal_ids(p, q, q[:2][::-1], p[:2][::-1])
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_array_equal(reverse, [1, 0])


def test_each_endpoint_uses_euclidean_radius():
    p = np.array([[0, 0]])
    ids, _ = reciprocal_ids(p, p, p, p + [2.5, 2.5])
    assert len(ids) == 0
    ids, _ = reciprocal_ids(p, p, p + [3, 0], p + [0, 3])
    assert len(ids) == 1


def test_mutual_filter_is_one_to_one():
    p = np.array([[0, 0], [0.1, 0]])
    ids, reverse = reciprocal_ids(p, p, p[:1], p[:1])
    np.testing.assert_array_equal(ids, [0])
    np.testing.assert_array_equal(reverse, [0])


def test_empty_and_invalid():
    empty = np.empty((0, 2))
    assert len(reciprocal_ids(empty, empty, empty, empty)[0]) == 0
    with pytest.raises(ValueError):
        reciprocal_ids([[np.nan, 0]], [[0, 0]], empty, empty)


def test_consistently_wrong_matches_can_survive():
    p = np.array([[0, 0], [10, 10]])
    wrong = p + 100
    ids, _ = reciprocal_ids(p, wrong, wrong, p)
    assert len(ids) == 2  # Repeatability must never be reported as accuracy.
