import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from aero_ir.registration.stereo_patch_evidence import (
    aggregate,
    patch_evidence,
    patches,
    summarize,
)


def test_bilinear_patch_centres_and_border_support():
    y, x = np.mgrid[:20, :30]
    image = 3 * x + 2 * y
    support = np.ones(image.shape, bool)
    values, valid = patches(image, support, np.array([[12.5, 9.5], [0.0, 0.0]]))
    assert valid.tolist() == [True, False]
    assert values[0, 40] == 56.5
    assert np.isnan(values[1]).all()
    support[10, 13] = False
    assert not patches(image, support, np.array([[12.5, 9.5]]))[1][0]


def test_fractional_patches_match_independent_scipy_interpolation():
    image = np.random.default_rng(4).uniform(0, 255, (32, 64))
    xy = np.array([[20.37, 15.61], [40.125, 20.25]])
    dy, dx = np.mgrid[-4:5, -4:5]
    actual, valid = patches(image, np.ones(image.shape, bool), xy)
    expected = np.stack(
        [
            map_coordinates(image, [dy.ravel() + y, dx.ravel() + x], order=1, prefilter=False)
            for x, y in xy
        ]
    )
    assert valid.all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_known_shift_is_preferred_and_brightness_scale_is_invariant():
    left = np.random.default_rng(3).uniform(10, 90, (32, 64))
    right = np.zeros_like(left)
    right[:, :-6] = left[:, 6:] * 1.5 + 7
    support = np.ones(left.shape, bool)
    xy = np.array([[30.0, 15.0], [40.0, 15.0]])
    ds = dict(lidar=np.full(2, 5.0), sgbm=np.full(2, 6.0), raft=np.full(2, 7.0))
    result = patch_evidence(left, right, support, support, xy, ds)
    np.testing.assert_allclose(result["ncc_sgbm"], 1.0, atol=1e-14)
    assert np.all(result["ncc_lidar"] < 0.5)
    assert summarize(result)["all"]["sgbm_minus_lidar"]["positive"] == 2


def test_flat_patch_does_not_become_perfect_correspondence():
    image = np.ones((32, 64)) * 12
    support = np.ones(image.shape, bool)
    xy = np.array([[30.0, 15.0]])
    result = patch_evidence(
        image, image, support, support, xy, {k: np.ones(1) for k in ("lidar", "sgbm", "raft")}
    )
    stats = summarize(result)["all"]
    assert stats["reference_points"] == 1 and stats["common_points"] == 0
    assert stats["lidar"]["patch_supported"] == 1
    assert stats["lidar"]["finite_ncc"] == 0


def test_missing_candidate_cannot_enter_common_subset_or_erase_reference():
    image = np.random.default_rng(2).uniform(0, 255, (32, 64))
    support = np.ones(image.shape, bool)
    result = patch_evidence(
        image,
        image,
        support,
        support,
        np.array([[30.0, 15.0]]),
        dict(lidar=np.ones(1), sgbm=np.ones(1), raft=np.array([np.nan])),
    )
    stats = summarize(result)
    assert stats["all"]["reference_points"] == 1
    assert stats["all"]["common_points"] == 0
    assert sum(v["reference_points"] for k, v in stats.items() if k != "all") == 1
    row = dict(sensor="thr", condition="paired_timed", strata=stats)
    combined = aggregate([row])[0]
    assert combined["common_frames"] == 0
    assert combined["raft_minus_sgbm"]["median_of_frame_median_delta"] is None


@pytest.mark.parametrize("xy", [np.array([[np.nan, 9.0]]), np.array([[12.0, np.inf]])])
def test_nonfinite_centres_remain_unavailable(xy):
    image = np.ones((32, 64))
    values, valid = patches(image, np.ones(image.shape, bool), xy)
    assert not valid.any() and np.isnan(values).all()


def test_invalid_disparity_inventory_rejected():
    with pytest.raises(ValueError, match="disparities"):
        patch_evidence(
            np.ones((10, 10)),
            np.ones((10, 10)),
            np.ones((10, 10), bool),
            np.ones((10, 10), bool),
            np.array([[5.0, 5.0]]),
            {},
        )
