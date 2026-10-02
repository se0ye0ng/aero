import numpy as np

from scripts.probe_roma_local_repair import local_prediction, repair


def test_affine_interpolation_and_no_extrapolation():
    x, y = np.meshgrid(np.linspace(0.3, 0.7, 12), np.linspace(0.3, 0.7, 12))
    source = np.c_[x.ravel(), y.ravel()]
    queries = np.array([[0.5, 0.5], [0.1, 0.1]])
    predicted = local_prediction(source, source * 1.1 + 0.01, queries)
    np.testing.assert_allclose(predicted[0], queries[0] * 1.1 + 0.01)
    assert np.isnan(predicted[1]).all()


def test_collinear_or_missing_support_abstains():
    source = np.c_[np.linspace(0.4, 0.6, 20), np.full(20, 0.5)]
    query = np.array([[0.5, 0.5]])
    assert np.isnan(local_prediction(source, source, query)).all()
    assert np.isnan(local_prediction(source[:3], source[:3], query)).all()


def test_bad_local_patch_repaired_from_image_grid_without_gt():
    def forward(p):
        out = p.copy()
        bad = np.linalg.norm(p - 50, axis=1) < 2
        out[bad] += 20
        return out

    query = np.array([[50.0, 50.0], [70.0, 70.0]])
    before, after, changed, _ = repair(forward, lambda p: p, query, (100, 100), (100, 100))
    np.testing.assert_allclose(before[0], [70, 70])
    np.testing.assert_allclose(after, query, atol=1e-8)
    np.testing.assert_array_equal(changed, [True, False])
