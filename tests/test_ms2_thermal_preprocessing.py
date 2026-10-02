import numpy as np
import pytest

from aero_ir.registration.ms2_image_reference import display_thermal
from scripts.probe_ms2_thermal_preprocessing import preprocess_pair


def test_original_window_is_exact_and_does_not_modify_raw_images():
    left = np.arange(3000, 6000, dtype=np.uint16).reshape(30, 100)
    right = left - 100
    original = left.copy()
    images, audit = preprocess_pair(left, right, "original_fixed")
    np.testing.assert_array_equal(images[0], display_thermal(left, (3308.0, 4974.0)))
    np.testing.assert_array_equal(left, original)
    assert audit[0]["window_dn"] == audit[1]["window_dn"] == [3308.0, 4974.0]


def test_independent_percentiles_remove_known_positive_affine_intensity_change():
    left = np.arange(1000, 2000, dtype=np.uint16).reshape(20, 50)
    right = 2 * left + 321
    images, audit = preprocess_pair(left, right, "independent_percentiles")
    np.testing.assert_array_equal(images[0], images[1])
    assert audit[0]["window_dn"] != audit[1]["window_dn"]


def test_shared_percentiles_preserve_relative_intensity_shift():
    left = np.arange(1000, 2000, dtype=np.uint16).reshape(20, 50)
    images, audit = preprocess_pair(left, left + 100, "pair_shared_percentiles")
    assert audit[0]["window_dn"] == audit[1]["window_dn"]
    assert np.median(images[1].astype(float) - images[0]) > 0


def test_flat_images_are_retained_without_invented_contrast():
    left = np.full((16, 16), 3500, np.uint16)
    images, audit = preprocess_pair(left, left + 50, "independent_percentiles")
    assert not images[0].any() and not images[1].any()
    assert all(a["flat_percentile_range"] for a in audit)


def test_unknown_preprocessing_is_rejected():
    image = np.ones((16, 16), np.uint16)
    with pytest.raises(ValueError, match="unknown"):
        preprocess_pair(image, image, "pick_best")


def test_non_sensor_image_is_rejected():
    image = np.ones((16, 16), np.float32)
    with pytest.raises(ValueError, match="uint16"):
        preprocess_pair(image, image, "original_fixed")
