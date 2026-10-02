import numpy as np
import pytest

from scripts.probe_match_geometry import (
    box_iou,
    domain_has_pole,
    fit_geometry,
    geometric_coverage,
    project,
    unique_reciprocal,
)


@pytest.mark.parametrize(
    "model,matrix",
    [
        ("similarity", [[1.05, -0.03, 4], [0.03, 1.05, 5], [0, 0, 1]]),
        ("affine", [[1.05, 0.03, 4], [0.02, 0.95, 5], [0, 0, 1]]),
        ("homography", [[1.05, 0.03, 4], [0.02, 0.95, 5], [0.0001, -0.00005, 1]]),
    ],
)
def test_known_transform_heldout(model, matrix):
    matrix = np.array(matrix, dtype=float)
    a = np.random.default_rng(23).uniform(30, 140, (60, 2))
    b = project(matrix, a)
    box = np.array([40, 40, 80, 80], dtype=float)
    domain = np.array([0, 0, 200, 200], dtype=float)
    result = fit_geometry(a, b, model, [box, box], [domain, domain])
    assert result["status"] == "fit"
    assert result["fit_count"] == result["check_count"] == 30
    for row in result["heldout_consistency_not_accuracy"].values():
        assert row["median_pixels"] < 0.001
        assert row["fraction_within_3px"] == 1


@pytest.mark.parametrize("model", ["similarity", "affine", "homography"])
def test_insufficient_and_collinear_no_identity_fallback(model):
    box = np.array([0, 0, 100, 100], dtype=float)
    empty = np.empty((0, 2))
    result = fit_geometry(empty, empty, model, [box, box], [box, box])
    assert result["status"] == "insufficient_matches" and "matrix" not in result
    points = np.c_[np.arange(30), np.arange(30)]
    result = fit_geometry(points, points, model, [box, box], [box, box])
    assert result["status"] == "degenerate_fit_support" and "matrix" not in result


def test_reciprocal_dedup_uses_joint_pairs_preserves_subpixel():
    a = np.array([[1.1, 3.1], [1.2, 3.2], [20, 30]], dtype=float)
    b = a + 4
    aa, bb = unique_reciprocal(a, b, b, a)
    assert len(aa) == 2
    assert np.allclose(aa[0], a[0])
    assert np.allclose(bb - aa, 4)
    aa, _ = unique_reciprocal(a, b, b + 20, a)
    assert len(aa) == 0


def test_projective_pole_rejected_for_box_and_domain():
    h = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -5]], dtype=float)
    box = np.array([0, 0, 10, 10], dtype=float)
    assert domain_has_pole(h, box)
    assert box_iou(h, box, box) is None


def test_numerical_coverage_not_match_count():
    box = np.array([0, 0, 100, 100], dtype=float)
    assert geometric_coverage(np.eye(3), box, box) == 1
    h = np.eye(3)
    h[0, 2] = 50
    assert geometric_coverage(h, box, box) == 0.5
    assert box_iou(np.eye(3), box, box) == 1


def test_nonfinite_matches_rejected():
    box = np.array([0, 0, 100, 100], dtype=float)
    a = np.full((20, 2), float("nan"))
    assert fit_geometry(a, a, "affine", [box, box], [box, box])["status"] == "nonfinite_matches"
    with pytest.raises(ValueError, match="finite Nx2"):
        unique_reciprocal(a, a, a, a)
