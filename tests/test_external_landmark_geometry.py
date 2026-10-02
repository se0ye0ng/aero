import numpy as np
import pytest

from scripts.analyze_external_landmark_geometry import cross_fit, fit_predict, folds


def test_cross_validation_partition_coverage():
    split = list(folds(34, 5, 0))
    assert sorted(np.concatenate([b for _, b in split]).tolist()) == list(range(34))
    for a, b in split:
        assert not set(a) & set(b)
        assert set(a) | set(b) == set(range(34))
    assert not np.array_equal(split[0][1], list(folds(34, 5, 1))[0][1])


@pytest.mark.parametrize("n,k", [(4, 2), (10, 1), (10, 11), (0, 5)])
def test_invalid_partition(n, k):
    with pytest.raises(ValueError):
        list(folds(n, k, 0))


@pytest.mark.parametrize("family", ["affine", "homography", "thin_plate_spline"])
def test_known_affine_recovered_out_of_fit(family):
    points = np.random.default_rng(23).uniform(10, 900, (40, 2))
    target = points @ np.array([[0.5, -0.02], [0.1, 0.6]]) + [5, 10]
    errors, partitions = cross_fit(points, target, (1000, 1000), (640, 640), family, 5, 0, 0)
    assert errors.max() < 0.001
    assert all(p["failure"] is None for p in partitions)


def test_held_out_target_cannot_influence_own_prediction():
    points = np.random.default_rng(0).uniform(0, 100, (20, 2))
    train, test = list(folds(20, 5, 0))[0]
    target = points + [2, 3]
    first = fit_predict(
        "thin_plate_spline", points[train], target[train], points[test], (200, 200), (200, 200), 0
    )
    target[test] += 1000
    second = fit_predict(
        "thin_plate_spline", points[train], target[train], points[test], (200, 200), (200, 200), 0
    )
    np.testing.assert_array_equal(first, second)


def test_collinear_failures_keep_full_denominator():
    points = np.column_stack((np.arange(20), np.arange(20))).astype(float)
    errors, records = cross_fit(points, points, (30, 30), (30, 30), "homography", 5, 0, 0)
    assert len(errors) == 20 and np.isinf(errors).all()
    assert all(r["failure"] for r in records)
