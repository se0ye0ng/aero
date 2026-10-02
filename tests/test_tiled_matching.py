import numpy as np
import pytest

from aero_ir.registration.tiled_matching import match_tiles, merge_exact_matches, prepare_tiles


def test_all_cartesian_tile_pairs_are_evaluated_and_restored():
    calls = []

    def fake(a, b):
        calls.append((a.shape, b.shape))
        return np.array([[100.0, 100.0]]), np.array([[100.0, 100.0]]), np.array([0.9]), {}

    p, q, c, flags = match_tiles(
        fake, np.zeros((288, 640), dtype=np.uint8), np.zeros((408, 640), dtype=np.uint8)
    )
    assert len(calls) == flags["tile_pairs"] == 81
    assert len(p) == len(q) == len(c) == 81
    assert len(np.unique(p, axis=0)) == 9 and len(np.unique(q, axis=0)) == 9


def test_empty_tile_outputs_remain_empty():
    def fake(a, b):
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0), {}

    p, q, c, flags = match_tiles(
        fake, np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8)
    )
    assert p.shape == q.shape == (0, 2) and c.shape == (0,)
    assert flags["tile_pairs"] == 81


@pytest.mark.parametrize("shape", [(288, 640), (408, 640), (289, 641)])
def test_tiles_cover_every_pixel_and_restore_centres(shape):
    image = np.zeros(shape, dtype=np.uint8)
    coverage = np.zeros(shape, dtype=int)
    tiles = prepare_tiles(image)
    assert len(tiles) == 9
    for t, patch in tiles:
        assert patch.shape == (t.input_height, t.input_width)
        assert not t.input_width % 8 and not t.input_height % 8
        coverage[t.y : t.y + t.height, t.x : t.x + t.width] += 1
        original = np.array([[t.x + 3.0, t.y + 4.0], [t.x + t.width - 3, t.y + t.height - 3]])
        scale = np.array([t.input_width / t.width, t.input_height / t.height])
        model = (original - [t.x, t.y] + 0.5) * scale - 0.5
        np.testing.assert_allclose(t.restore(model), original, atol=1e-10)
    assert (coverage >= 1).all()


def test_deduplicate_only_identical_endpoint_pairs():
    p = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.01, 2.0]])
    q = np.array([[3.0, 4.0], [3.0, 4.0], [3.1, 4.0], [3.0, 4.0]])
    p2, q2, c2, keep = merge_exact_matches(p, q, [0.2, 0.9, 0.8, 0.7])
    np.testing.assert_array_equal(keep, [1, 2, 3])
    np.testing.assert_array_equal(p2, p[keep])
    np.testing.assert_array_equal(q2, q[keep])
    np.testing.assert_array_equal(c2, [0.9, 0.8, 0.7])


def test_empty_matches_remain_empty():
    p, q, c, ids = merge_exact_matches(np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    assert p.shape == q.shape == (0, 2) and c.shape == ids.shape == (0,)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        prepare_tiles(np.zeros((100, 100)))
    with pytest.raises(ValueError):
        prepare_tiles(np.zeros((100, 100), dtype=np.uint8), 638)
    with pytest.raises(ValueError):
        merge_exact_matches([[np.nan, 0]], [[0, 0]], [1])
