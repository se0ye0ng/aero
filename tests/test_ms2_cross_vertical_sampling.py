"""Check the fixed projection score, including its equal-depth ambiguity."""

import numpy as np

from aero_ir.registration.temporal_stereo import rectification
from scripts.analyze_ms2_cross_stereo import expected_disparity
from scripts.probe_ms2_cross_vertical_sampling import evaluate


def fixture():
    k = np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]])
    t = np.eye(4)
    t[0, 3] = -0.3
    geometry = rectification(k, k, t, (256, 640))
    reference = dict(
        projected_xy=np.array([[300.0, 120.0], [350.0, 140.0], [400.0, 160.0]]),
        expected_z=np.full(3, 10.0),
        reference_supported=np.array([True, False, True]),
        source_depth_m=np.full(3, 10.0),
    )
    disparity = expected_disparity(reference["projected_xy"], reference["expected_z"], geometry)[0]
    stereo = dict(disparity=np.full((256, 640), disparity), valid=np.ones((256, 640), bool))
    return stereo, geometry, reference


def test_exact_depth_and_fixed_reference():
    stereo, geometry, reference = fixture()
    z, errors, scores = evaluate(stereo, geometry, reference, 100.0)
    np.testing.assert_allclose(z[[0, 2]], 10.0, atol=1e-12)
    np.testing.assert_allclose(errors[[0, 2]], 0.0, atol=1e-12)
    assert np.isnan(z[1]) and np.isnan(errors[1])
    assert scores["strata"]["all"]["reference_points"] == 2
    assert scores["strata"]["all"]["supported"] == 2


def test_equal_depth_wrong_coordinates_are_not_identifiable():
    stereo, geometry, reference = fixture()
    reference["projected_xy"] += np.array([40.0, 20.0])
    _, errors, scores = evaluate(stereo, geometry, reference, 100.0)
    np.testing.assert_allclose(errors[[0, 2]], 0.0, atol=1e-12)
    assert scores["strata"]["all"]["supported"] == 2


def test_missing_depth_stays_in_denominator():
    stereo, geometry, reference = fixture()
    stereo["valid"][:] = False
    _, errors, scores = evaluate(stereo, geometry, reference, 100.0)
    assert np.isnan(errors).all()
    assert scores["strata"]["all"]["reference_points"] == 2
    assert scores["strata"]["all"]["supported"] == 0
    assert scores["strata"]["all"]["fraction_reference_within_px"]["3.0"] == 0.0


def test_error_uses_equivalent_focal_scale():
    stereo, geometry, reference = fixture()
    stereo["disparity"] += 0.5
    _, errors, _ = evaluate(stereo, geometry, reference, 200.0)
    np.testing.assert_allclose(errors[[0, 2]], 0.5 * 200.0 / geometry["p1"][0, 0], atol=1e-12)
