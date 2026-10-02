import numpy as np
import pytest

from aero_ir.registration.roma_dense import dense_predictor, split_dense


def grid(w, h):
    x, y = np.meshgrid(2 * (np.arange(w) + 0.5) / w - 1, 2 * (np.arange(h) + 0.5) / h - 1)
    return np.stack([x, y], axis=-1)


def test_symmetric_identity_and_header_offsets():
    g = grid(10, 8)
    warp = np.concatenate([np.concatenate([g, g], -1)] * 2, axis=1)
    f, r = split_dense(warp, np.ones((8, 20)))
    for field, c in (f, r):
        predict = dense_predictor(field, c, (10, 8), (10, 8), 72, 104)
        np.testing.assert_allclose(predict([[2, 75], [5, 78]]), [[2, 107], [5, 110]])


def test_known_translation_and_confidence():
    f = grid(10, 8) + [0.4, -0.25]
    p = dense_predictor(f, np.ones((8, 10)), (10, 8), (10, 8))
    np.testing.assert_allclose(p([[3.25, 4.5]]), [[5.25, 3.5]])
    assert np.isnan(p([[-1, 2], [9, 2]])).all()
    p = dense_predictor(f, np.full((8, 10), 0.49), (10, 8), (10, 8))
    assert np.isnan(p([[3, 4]])).all()


def test_wrong_source_grid_rejected():
    with pytest.raises(AssertionError):
        split_dense(np.zeros((8, 20, 4)), np.ones((8, 20)))
