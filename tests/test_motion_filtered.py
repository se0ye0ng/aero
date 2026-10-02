import numpy as np
import pytest

from scripts.probe_antiuav_motion_filtered import mask_membership


def test_header_restore_and_no_border_clipping():
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, 2] = True
    mask[3, 4] = True
    points = [[2, 72], [4, 75], [2, 71.9], [5, 75], [4, 75.1]]
    np.testing.assert_array_equal(
        mask_membership(points, mask, 72), [True, True, False, False, False]
    )


def test_nearest_pixel_center():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    assert mask_membership([[0.51, 0.51]], mask, 0)[0]
    assert not mask_membership([[0.49, 0.49]], mask, 0)[0]


def test_invalid_and_empty():
    mask = np.ones((3, 3), dtype=bool)
    assert len(mask_membership(np.empty((0, 2)), mask, 0)) == 0
    with pytest.raises(ValueError):
        mask_membership([[np.nan, 0]], mask, 0)
