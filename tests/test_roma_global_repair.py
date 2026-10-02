import numpy as np

from scripts.probe_roma_global_repair import robust_affine


def test_affine_recovers_without_query_destinations():
    rng = np.random.default_rng(0)
    source = rng.uniform(10, 500, (100, 2))
    target = source * 1.1 + [5, 7]
    target[:15] += 50
    query = np.array([[50.0, 90.0]])
    result, info = robust_affine(source, target, query)
    assert info["accepted"]
    assert info["inliers"] == 85
    np.testing.assert_allclose(result, query * 1.1 + [5, 7], atol=1e-4)


def test_reflection_and_insufficient_controls_abstain():
    source = np.random.default_rng(0).uniform(10, 500, (100, 2))
    result, info = robust_affine(source, source * [-1, 1], source[:1])
    assert not info["accepted"]
    assert np.isnan(result).all()
    result, info = robust_affine(source[:5], source[:5], source[:1])
    assert not info["accepted"]
