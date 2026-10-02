import numpy as np
import pytest

from scripts.probe_external_local_warp import fit_image_warp


def config():
    return dict(
        neighbors=12,
        trim_fraction=0.75,
        trim_iterations=4,
        agreement_min_thermal_pixels=2.0,
        agreement_thermal_long_side_fraction=0.005,
        control_grid=12,
        minimum_controls=6,
        tps_smoothing=0.0001,
    )


def matches():
    x, y = np.meshgrid(np.linspace(10, 90, 9), np.linspace(10, 90, 9))
    rgb = np.column_stack((x.ravel(), y.ravel()))
    return rgb, rgb * 0.8 + 5, np.ones(len(rgb))


def test_affine_field_and_extrapolation_rejection():
    rgb, thermal, confidence = matches()
    predict, info = fit_image_warp(rgb, thermal, confidence, (100, 100), (100, 100), config())
    assert info["status"] == "fit"
    np.testing.assert_allclose(predict([[35, 35], [42, 62]]), [[33, 33], [38.6, 54.6]])
    assert np.isnan(predict([[0, 0]])).all()


def test_own_bad_match_cannot_validate_itself():
    rgb, thermal, confidence = matches()
    thermal[40] += [40, 30]
    predict, info = fit_image_warp(rgb, thermal, confidence, (100, 100), (100, 100), config())
    assert 40 not in info["control_match_indices"]
    np.testing.assert_allclose(predict([rgb[40]]), [rgb[40] * 0.8 + 5], atol=0.1)


def test_insufficient_matches_fail_explicitly():
    predict, info = fit_image_warp(
        np.zeros((3, 2)), np.zeros((3, 2)), np.ones(3), (100, 100), (100, 100), config()
    )
    assert predict is None and info["status"] == "insufficient_matches"


def test_nonfinite_match_rejected():
    rgb, thermal, confidence = matches()
    confidence[0] = np.nan
    with pytest.raises(ValueError):
        fit_image_warp(rgb, thermal, confidence, (100, 100), (100, 100), config())
