import numpy as np
import pytest

from aero_ir.registration.stereo_anchor_profile import (
    aggregate,
    known_shift_control,
    summarize,
    vertical_profile,
)
from aero_ir.registration.stereo_patch_profile import OFFSETS, axis_profiles


@pytest.mark.parametrize("shift", [0, 1])
def test_natural_texture_control_recovers_zero_and_positive_vertical_shift(shift):
    image = np.random.default_rng(0).uniform(0, 255, (32, 64))
    scores, stats = known_shift_control(
        image, np.ones(image.shape, bool), np.array([[30.25, 15.5], [40.75, 16.25]]), shift
    )
    assert stats["ok"] and stats["complete_profiles"] == 2
    assert stats["peaks"]["median_unique_offset_px"] == shift
    np.testing.assert_allclose(scores[:, np.flatnonzero(OFFSETS == shift)[0]], 1.0, atol=1e-12)


def test_vertical_only_calculation_matches_frozen_axis_helper():
    image = np.random.default_rng(1).uniform(0, 255, (32, 64))
    mask = np.ones(image.shape, bool)
    xy, d = np.array([[30.25, 15.5]]), np.array([5.5])
    expected = axis_profiles(image, image, mask, mask, xy, d)["ncc_profile_y"]
    np.testing.assert_allclose(
        vertical_profile(image, image, mask, mask, xy, d),
        expected,
        atol=1e-12,
        rtol=0,
        equal_nan=True,
    )


def test_common_points_identical_across_anchors_and_missing_not_deleted():
    profiles = {k: -np.abs(OFFSETS)[None].repeat(2, axis=0) for k in ("lidar", "sgbm", "raft")}
    profiles["raft"][0, 0] = np.nan
    stats = summarize(profiles, np.array([10.0, 10.0]))
    assert stats["all"]["reference_points"] == 2
    assert stats["all"]["common_points"] == 1
    assert all(v["common"]["points"] == 1 for v in stats["all"]["anchors"].values())
    result = aggregate([dict(sensor="thr", condition="paired", strata=stats)])[0]
    assert result["reference_points"] == 2 and result["common_points"] == 1


def test_flat_image_is_uninformative_control_not_success():
    image = np.ones((32, 64))
    _, stats = known_shift_control(image, np.ones(image.shape, bool), np.array([[30.0, 15.0]]), 0)
    assert not stats["ok"] and stats["complete_profiles"] == 0


def test_no_complete_points_has_no_peak_estimate():
    profiles = {k: np.full((1, 13), np.nan) for k in ("lidar", "sgbm", "raft")}
    stats = summarize(profiles, np.array([np.nan]))
    result = aggregate([dict(sensor="thr", condition="paired", strata=stats)])[0]
    assert result["reference_points"] == 1
    assert all(v["median_of_common_frame_peak_offsets"] is None for v in result["anchors"].values())
