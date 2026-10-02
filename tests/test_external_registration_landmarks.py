import hashlib

import numpy as np
import pytest

from scripts.fetch_registration_landmarks import select_samples, verify_git_blob
from scripts.probe_external_registration_landmarks import (
    estimate,
    landmark_errors,
    resize_gray,
    summarize,
    to_native,
    validate_landmarks,
)


def test_numeric_selection_is_predeclared_not_score_based():
    entries = [dict(type="tree", name=n, path=f"Scene/{n}") for n in ("10", "2", "1")]
    assert [e["name"] for e in select_samples(entries, 2)] == ["1", "2"]
    with pytest.raises(ValueError):
        select_samples(entries, 4)


def test_blob_integrity():
    data = b"abc"
    digest = hashlib.sha1(b"blob 3\0abc").hexdigest()
    verify_git_blob(data, digest)
    with pytest.raises(ValueError):
        verify_git_blob(b"abd", digest)


def test_pixel_center_roundtrip():
    small, scale = resize_gray(np.zeros((3000, 4000), dtype=np.uint8), 640)
    assert small.shape == (480, 640)
    points = np.array([[0.0, 0.0], [639.0, 479.0], [31.5, 9.1]])
    np.testing.assert_allclose((to_native(points, scale) + 0.5) * scale - 0.5, points, atol=1e-12)


def test_known_transform_forward_and_inverse():
    points = np.array([[1.0, 2.0], [4.0, 8.0], [30.0, 40.0], [50.0, 10.0]])
    h = np.array([[2.0, 0.0, 5.0], [0.0, 3.0, -4.0], [0.0, 0.0, 1.0]])
    rgb = points * [2.0, 3.0] + [5.0, -4.0]
    for errors in landmark_errors(h, points, rgb):
        np.testing.assert_allclose(errors, 0.0, atol=1e-12)


def test_invalid_homography_counts_all_landmarks_as_failures():
    points = np.ones((7, 2))
    for matrix in (None, np.zeros((3, 3))):
        errors = landmark_errors(matrix, points, points)[1]
        summary = summarize(errors, [1.0, 3.0])
        assert summary["landmarks"] == summary["unprojectable"] == 7
        assert summary["pck_all_landmarks"] == {"1.0": 0.0, "3.0": 0.0}
        assert summary["conditional_finite_median"] is None


def test_partial_failures_not_dropped():
    summary = summarize([0.0, 2.0, np.inf, 10.0], [1.0, 3.0])
    assert summary["pck_all_landmarks"] == {"1.0": 0.25, "3.0": 0.5}


@pytest.mark.parametrize(
    "points", [np.ones((3, 2)), np.ones((4, 3)), np.full((4, 2), np.nan), np.full((4, 2), 100.0)]
)
def test_invalid_reference_coordinates(points):
    with pytest.raises(ValueError):
        validate_landmarks(points, (10, 10))


def test_image_match_only_estimator():
    config = dict(
        ransac_threshold_native_rgb_pixels=1.0, ransac_confidence=0.999, ransac_max_iterations=1000
    )
    points = np.array([[0.0, 0.0], [0.0, 40.0], [40.0, 0.0], [40.0, 40.0], [20.0, 10.0]])
    matrix, inliers = estimate(points, points + [4.0, -3.0], config, 0)
    assert inliers == 5
    np.testing.assert_allclose(
        matrix, [[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]], atol=1e-8
    )
    assert estimate(points[:3], points[:3], config, 0) == (None, 0)
