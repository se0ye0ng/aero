import numpy as np
import pytest

from aero_ir.registration.local_consensus_warp import (
    farthest_controls,
    fit_image_warp,
    select_controls,
)
from scripts.probe_external_local_warp import fit_image_warp as previous_fit
from tests.test_external_local_warp import config, matches


def test_under_budget_keeps_every_point_even_in_one_cell():
    points = np.array([[10.0, 10.0], [10.2, 10.1], [10.1, 10.2]])
    settings = dict(control_grid=12, maximum_controls=144)
    accepted = np.arange(3)
    assert select_controls(points, (1000, 1000), accepted, settings, "grid") == [0]
    assert select_controls(points, (1000, 1000), accepted, settings, "fps") == [0, 1, 2]


def test_farthest_sampling_is_bounded_deterministic_and_unique():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    a = farthest_controls(points, 3)
    assert a.tolist() == [0, 4, 2]
    np.testing.assert_array_equal(a, farthest_controls(points, 3))
    assert len(set(a)) == 3
    assert farthest_controls(np.empty((0, 2)), 144).size == 0


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_control_budget(limit):
    with pytest.raises(ValueError):
        farthest_controls(np.ones((3, 2)), limit)


def test_preserved_grid_predictor_parity():
    rgb, thermal, confidence = matches()
    settings = dict(config(), maximum_controls=144)
    old, old_info = previous_fit(rgb, thermal, confidence, (100, 100), (100, 100), settings)
    new, info = fit_image_warp(rgb, thermal, confidence, (100, 100), (100, 100), settings, "grid")
    assert old_info == info
    np.testing.assert_array_equal(old(rgb), new(rgb))


def test_fps_keeps_affine_accuracy_and_rejects_extrapolation():
    rgb, thermal, confidence = matches()
    predict, info = fit_image_warp(
        rgb,
        thermal,
        confidence,
        (100, 100),
        (100, 100),
        dict(config(), maximum_controls=144),
        "fps",
    )
    assert info["controls"] == len(rgb)
    np.testing.assert_allclose(predict([[40.0, 40.0]]), [[37.0, 37.0]], atol=1e-8)
    assert np.isnan(predict([[0.0, 0.0]])).all()
