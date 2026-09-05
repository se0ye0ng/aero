"""RFS components must respond to the physics they claim to measure."""

import numpy as np
import pytest

from aero_ir.rfs import stats


def _scene(delta=20.0, noise=1.0, size=128, box=(50, 50, 12, 12), seed=0):
    rng = np.random.default_rng(seed)
    img = 100.0 + rng.normal(0, noise, (size, size))
    x, y, w, h = box
    img[y : y + h, x : x + w] += delta
    return img, [box]


def test_delta_recovers_injected_contrast():
    img, boxes = _scene(delta=20.0, noise=0.5)
    d = stats.target_background_delta(img, boxes)
    assert d.size == 1
    assert d[0] == pytest.approx(20.0, abs=2.0)


def test_polarity_sign_follows_contrast_direction():
    hot, boxes = _scene(delta=15.0)
    cold, _ = _scene(delta=-15.0)
    assert stats.thermal_polarity(hot, boxes)[0] == 1
    assert stats.thermal_polarity(cold, boxes)[0] == -1


def test_snr_falls_when_noise_rises():
    quiet, boxes = _scene(delta=10.0, noise=0.5)
    loud, _ = _scene(delta=10.0, noise=5.0)
    assert stats.target_snr(quiet, boxes)[0] > stats.target_snr(loud, boxes)[0]


def test_blurred_image_loses_high_frequency_power():
    from scipy import ndimage

    img, _ = _scene(noise=2.0)
    sharp = stats.radial_power_spectrum(img)
    blurred = stats.radial_power_spectrum(ndimage.gaussian_filter(img, 2.0))
    assert blurred[-8:].mean() < sharp[-8:].mean()


def test_column_structure_detects_injected_column_noise():
    img, _ = _scene(noise=0.5)
    clean = stats.column_structure_energy(img)[0]
    rng = np.random.default_rng(1)
    img_fpn = img + rng.normal(0, 2.0, (1, img.shape[1]))
    assert stats.column_structure_energy(img_fpn)[0] > clean


def test_target_pixel_area():
    assert stats.target_pixel_area([(0, 0, 4, 5)])[0] == pytest.approx(20.0)


def test_spectral_bins_keep_their_coordinate_identity():
    class Cfg:
        components = ["R4", "R5", "R7"]
        annulus_dilation_px = 8
        highpass_sigma_px = 1.5

    images_and_boxes = [_scene(seed=seed) for seed in range(4)]
    values = stats.image_set_statistics(
        [item[0] for item in images_and_boxes],
        [item[1] for item in images_and_boxes],
        Cfg(),
    )
    assert values["R4"].shape == (4, 64)
    assert values["R5"].shape == (4, 32)
    assert values["R7"].shape == (4, 3)
