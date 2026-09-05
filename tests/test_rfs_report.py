"""End-to-end RFS behaviour: a radiometrically degraded set must score further from real.

This is the smallest test that would fail if RFS stopped measuring what it claims to measure,
and it doubles as the worked example of what a report looks like.
"""

import numpy as np
import pytest
from scipy import ndimage

from aero_ir.rfs import compute_rfs


class _Cfg:
    components = ["R1", "R3", "R4", "R5", "R6", "R8"]
    distance = "wasserstein"
    weights = "uniform"
    annulus_dilation_px = 8
    highpass_sigma_px = 1.5


def _make_set(n, delta, noise, blur=0.0, seed=0):
    rng = np.random.default_rng(seed)
    images, boxes = [], []
    for _ in range(n):
        img = 100.0 + rng.normal(0, noise, (96, 96))
        img[40:50, 40:50] += delta
        if blur:
            img = ndimage.gaussian_filter(img, blur)
        images.append(img)
        boxes.append([(40, 40, 10, 10)])
    return images, boxes


def test_degraded_set_scores_worse_than_faithful_set():
    real = _make_set(24, delta=20.0, noise=2.0, seed=0)
    faithful = _make_set(24, delta=20.0, noise=2.0, seed=1)
    # compressed contrast, no sensor noise, over-smoothed - the classic generated-IR signature
    degraded = _make_set(24, delta=6.0, noise=0.2, blur=2.0, seed=2)

    r_faithful = compute_rfs(real, faithful, _Cfg())
    r_degraded = compute_rfs(real, degraded, _Cfg())

    assert r_degraded.scalar > r_faithful.scalar
    assert r_faithful.scalar < 5.0, "a faithful set should sit near the real sampling floor"


def test_report_names_the_component_that_failed():
    real = _make_set(20, delta=20.0, noise=2.0, seed=0)
    too_clean = _make_set(20, delta=20.0, noise=0.1, seed=3)
    report = compute_rfs(real, too_clean, _Cfg())
    # Removing sensor noise shows up in two places, and both are correct diagnoses:
    # R5 (the noise PSD floor drops) and R3 (target SNR inflates because sigma_bg -> 0).
    # The point of a diagnostic vector rather than a scalar is that it says which.
    assert {"R5", "R3"} & {name for name, _ in report.worst(2)}


def test_identical_distributions_sit_near_the_floor():
    a = _make_set(24, delta=15.0, noise=1.5, seed=10)
    b = _make_set(24, delta=15.0, noise=1.5, seed=11)
    report = compute_rfs(a, b, _Cfg())
    assert report.scalar == pytest.approx(1.0, abs=1.5)


def test_zero_floor_does_not_hide_polarity_inversion():
    class PolarityCfg(_Cfg):
        components = ["R2"]

    real = _make_set(8, delta=20.0, noise=0.0)
    inverted = _make_set(8, delta=-20.0, noise=0.0)
    report = compute_rfs(real, inverted, PolarityCfg())
    assert np.isinf(report.per_component["R2"])
    assert np.isinf(report.scalar)
