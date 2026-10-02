import numpy as np
import pytest

from aero_ir.registration.roma_coordinates import normalized_to_centres, rgb_image


@pytest.mark.parametrize("size,top", [((640, 288), 72), ((640, 408), 104), ((13, 19), 0)])
def test_grid_maps_to_pixel_centres(size, top):
    w, h = size
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xy = np.c_[xx.ravel(), yy.ravel()]
    norm = 2 * (xy + 0.5) / [w, h] - 1
    np.testing.assert_allclose(normalized_to_centres(norm, size, top), xy + [0, top], atol=1e-12)


def test_no_clipping_at_normalized_edges():
    out = normalized_to_centres([[-1, -1], [1, 1]], (10, 20))
    np.testing.assert_array_equal(out, [[-0.5, -0.5], [9.5, 19.5]])


def test_rgb_order_and_grayscale_replication():
    color = np.array([[[255, 0, 17]]], dtype=np.uint8)
    np.testing.assert_array_equal(np.array(rgb_image(color)), color)
    gray = np.array([[12, 39]], dtype=np.uint8)
    np.testing.assert_array_equal(np.array(rgb_image(gray)), np.repeat(gray[..., None], 3, 2))


def test_invalid_values():
    with pytest.raises(ValueError):
        normalized_to_centres([[np.nan, 0]], (2, 2))
    with pytest.raises(ValueError):
        rgb_image(np.ones((2, 2), dtype=float))
