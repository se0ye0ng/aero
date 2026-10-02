import numpy as np
import pytest

from aero_ir.registration.stereo_patch_profile import (
    OFFSETS,
    aggregate,
    axis_profiles,
    peak_statistics,
    summarize,
)


def test_known_horizontal_error_has_correct_right_coordinate_sign():
    left = np.random.default_rng(0).uniform(0, 255, (32, 64))
    right = np.zeros_like(left)
    right[:, :-6] = left[:, 6:]
    mask = np.ones(left.shape, bool)
    result = axis_profiles(left, right, mask, mask, np.array([[30.0, 16.0]]), np.array([5.0]))
    stats = summarize(result, np.array([30.0]))["all"]
    assert stats["x"]["median_unique_offset_px"] == -1
    assert result["ncc_profile_x"][0, np.flatnonzero(OFFSETS == -1)[0]] == pytest.approx(1.0)


def test_known_vertical_error_is_separate_from_horizontal():
    left = np.random.default_rng(1).uniform(0, 255, (32, 64))
    right = np.zeros_like(left)
    right[2:, :-6] = left[:-2, 6:]
    mask = np.ones(left.shape, bool)
    result = axis_profiles(left, right, mask, mask, np.array([[30.0, 16.0]]), np.array([6.0]))
    assert summarize(result, np.array([30.0]))["all"]["y"]["median_unique_offset_px"] == 2.0


def test_flat_peak_is_ambiguous_not_first_offset():
    result = peak_statistics(np.ones((2, 13)), np.ones(2, bool))
    assert result["tied_peak_points"] == 2
    assert result["median_unique_offset_px"] is None
    assert sum(result["offset_counts"].values()) == 0
    assert result["boundary_peak_points"] == 2


def test_peak_rival_is_at_least_one_pixel_away():
    values = -np.abs(OFFSETS)[None]
    result = peak_statistics(values, np.ones(1, bool))
    assert result["median_separated_peak_gap"] == 1.0
    assert result["median_peak_gain"] == 0.0


def test_incomplete_profile_keeps_reference_and_cannot_choose_easier_support():
    values = np.zeros((2, 13))
    values[0, 0] = np.nan
    stats = summarize(
        dict(ncc_profile_x=values, ncc_profile_y=np.zeros((2, 13))), np.array([10.0, np.nan])
    )
    assert stats["all"]["reference_points"] == 2
    assert stats["all"]["complete_profiles"] == 1
    result = aggregate([dict(sensor="thr", condition="paired", strata=stats)])[0]
    assert result["reference_points"] == 2 and result["complete_profiles"] == 1


def test_boundary_peak_is_counted_and_not_extrapolated():
    stats = peak_statistics(OFFSETS[None], np.ones(1, bool))
    assert stats["boundary_peak_points"] == 1
    assert stats["median_unique_offset_px"] == 3.0


def test_nonfinite_eligible_profile_rejected():
    with pytest.raises(ValueError, match="all offsets"):
        peak_statistics(np.full((1, 13), np.nan), np.ones(1, bool))
