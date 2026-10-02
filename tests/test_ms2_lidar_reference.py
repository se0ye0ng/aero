import numpy as np
import pytest

from scripts.analyze_ms2_lidar_reference import (
    encoding_sensitivity,
    reference_strata,
    sensitivity_summary,
)


def test_sparse_depth_partition_retains_every_measured_point():
    depth = np.full((12, 12), np.nan)
    depth[0, 0] = 10
    depth[5, 5:8] = [10, 10, 20]
    strata = reference_strata(depth, 120.0, 2)
    assert strata["all"].sum() == 4
    assert strata["sparse_neighbourhood"].sum() == 1
    assert strata["local_range_gt_1px"].sum() == 3
    np.testing.assert_array_equal(
        sum(v.astype(int) for k, v in strata.items() if k != "all"), np.ones(4, int)
    )


def test_constant_depth_and_border_do_not_invent_returns():
    depth = np.full((6, 6), np.nan)
    depth[0, :3] = 10
    strata = reference_strata(depth, 120.0, 2)
    assert strata["local_range_le_1px"].sum() == 3
    depth[0, 2] = np.nan
    assert reference_strata(depth, 120.0, 2)["sparse_neighbourhood"].sum() == 2


def test_empty_map_is_not_a_perfect_score():
    strata = reference_strata(np.full((3, 3), np.nan), 120.0, 4)
    assert not any(v.size for v in strata.values())
    summary = sensitivity_summary(np.array([]), np.array([]))
    assert summary["max_change_px"] is None


def test_encoding_sensitivity_matches_inverse_depth_formula():
    p1 = np.c_[np.diag([400.0, 400.0, 1.0]), np.zeros(3)]
    p2 = p1.copy()
    p2[0, 3] = -120
    geometry = dict(left_k=p1[:, :3], r1=np.eye(3), p1=p1, p2=p2)
    z = np.array([10.0, 20.0, 1 / 256])
    result = encoding_sensitivity(np.zeros((3, 2)), z, geometry, 200.0)
    np.testing.assert_allclose(result[:2], 60 * (1 / (z[:2] - 1 / 256) - 1 / z[:2]))
    assert np.isnan(result[2])


def test_missing_estimates_and_sensitivities_not_silently_counted():
    result = sensitivity_summary(
        np.array([0.1, np.nan, 0.2, 0.4]), np.array([0.1, 0.1, np.nan, 0.1])
    )
    assert result["reference_points"] == 4
    assert result["supported"] == 3
    assert result["supported_with_finite_sensitivity"] == 2
    assert result["supported_residual_within_one_dn_change"] == 1


@pytest.mark.parametrize("radius,coefficient", [(1, 120), (2, 0), (4, np.nan)])
def test_invalid_configuration_rejected(radius, coefficient):
    with pytest.raises(ValueError):
        reference_strata(np.ones((3, 3)), coefficient, radius)
