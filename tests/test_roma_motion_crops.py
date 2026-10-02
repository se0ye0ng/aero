import numpy as np

from scripts.audit_roma_crop_support import covers_box_corners
from scripts.prepare_roma_motion_crops import crop_bounds, crop_identity, select_component


def test_motion_selection_does_not_require_boxes():
    mask = np.zeros((30, 40), dtype=bool)
    mask[10:12, 20:22] = True
    mask[20:22, 30:32] = True
    scores = mask.astype(float)
    scores[20:22, 30:32] *= 3
    selected = select_component(scores, mask)
    assert selected[20:22, 30:32].all()
    assert crop_bounds(selected) == [28, 18, 34, 24]


def test_empty_and_border_components():
    mask = np.zeros((20, 30), dtype=bool)
    assert crop_bounds(mask) is None
    mask[:2, :2] = True
    assert crop_bounds(mask) == [0, 0, 4, 4]


def test_coordinate_roundtrip_and_support():
    a, b = [10, 20, 30, 40], [30, 40, 70, 80]
    forward = crop_identity(a, b, 72, 104)
    reverse = crop_identity(b, a, 104, 72)
    p = np.array([[15.0, 100.0]])
    np.testing.assert_allclose(reverse(forward(p)), p)
    assert np.isnan(forward([[0, 0]])).all()


def test_frozen_corner_support_convention():
    assert covers_box_corners([0, 0, 10, 10], [0, 72, 9, 81], 72)
    assert not covers_box_corners([0, 0, 10, 10], [0, 72, 10, 82], 72)
    assert not covers_box_corners(None, [0, 72, 9, 81], 72)
