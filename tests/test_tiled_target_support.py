import numpy as np
import pytest

from scripts.analyze_tiled_target_support import in_box, membership


def test_box_edges_half_open():
    points = [[1, 1], [3, 3], [2.99, 2.99], [0, 1]]
    np.testing.assert_array_equal(in_box(points, [1, 1, 3, 3]), [True, False, True, False])


def test_membership_is_not_semantic_accuracy():
    p = np.array([[1, 1], [2, 2], [10, 10]])
    q = np.array([[2, 2], [10, 10], [1, 1]])
    stats = membership(p, q, [[0, 0, 3, 3], [0, 0, 3, 3]])
    assert stats == dict(total=3, source_box=2, target_box=2, both_boxes=1, source_box_to_outside=1)


def test_empty_and_invalid_box():
    assert membership(np.empty((0, 2)), np.empty((0, 2)), [[0, 0, 1, 1]] * 2)["total"] == 0
    with pytest.raises(ValueError):
        in_box([[0, 0]], [1, 1, 0, 0])
