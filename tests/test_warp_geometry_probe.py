import numpy as np

from aero_ir.registration.warp_geometry_probe import sample_geometry


def test_known_affine_and_inverse():
    a, s = sample_geometry(
        lambda x: x * [2.0, 3.0] + [4.0, 5.0], lambda y: (y - [4.0, 5.0]) / [2.0, 3.0], (64, 64)
    )
    np.testing.assert_allclose(a["determinant"], 6.0)
    np.testing.assert_allclose(a["singular_values"], np.tile([3.0, 2.0], (16, 1)))
    assert s["grid_points"] == s["jacobian_supported"] == s["cycle_supported"] == 16
    assert s["sampled_nonpositive_determinants"] == 0
    assert s["cycle_p95_native_source_px"] == 0.0


def test_reflection_is_not_passed_by_perfect_cycle():
    _, s = sample_geometry(lambda x: -x, lambda x: -x, (64, 64))
    # Two-axis reversal has positive determinant despite rotating by 180 degrees.
    assert s["sampled_nonpositive_determinants"] == 0
    _, s = sample_geometry(lambda x: x * [-1.0, 1.0], lambda x: x * [-1.0, 1.0], (64, 64))
    assert s["sampled_nonpositive_determinants"] == 16
    assert s["cycle_p95_native_source_px"] == 0.0


def test_missing_warp_retains_full_grid():
    a, s = sample_geometry(None, None, (64, 64))
    assert s["grid_points"] == 16 and s["forward_supported"] == 0
    assert s["cycle_median_native_source_px"] is None
    assert s["fraction_all_grid_cycle_within_3px"] == 0.0
    assert np.isnan(a["determinant"]).all()


def test_partial_support_not_removed_from_denominator():
    def half(x):
        y = x.copy()
        y[x[:, 0] >= 20] = np.nan
        return y

    _, s = sample_geometry(half, lambda x: x, (32, 32))
    assert s["grid_points"] == 4 and s["jacobian_supported"] == 2
    assert s["fraction_all_grid_cycle_within_3px"] == 0.5
