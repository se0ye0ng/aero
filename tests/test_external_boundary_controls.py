import numpy as np
import yaml

from scripts.probe_external_boundary_controls import accepted_matches, augment_boundary
from scripts.probe_external_local_warp import fit_image_warp


def test_boundary_addition_preserves_original_order_and_never_adds_rejected():
    points = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [20, 20]])
    ids = augment_boundary(points, np.arange(5), np.array([4, 1]))
    np.testing.assert_array_equal(ids[:2], [4, 1])
    assert set(ids) == set(range(5)) and len(ids) == len(set(ids))


def test_consensus_matches_frozen_implementation():
    from pathlib import Path

    config = yaml.safe_load(
        Path("configs/experiment/registration_external_local_warp_cpu.yaml").read_text()
    )
    rng = np.random.default_rng(0)
    rgb = rng.uniform(20, 150, (80, 2))
    thermal = rgb * 0.5 + 5
    confidence = rng.uniform(size=80)
    _, info = fit_image_warp(rgb, thermal, confidence, (200, 200), (100, 100), config)
    accepted, grid = accepted_matches(rgb, thermal, confidence, (200, 200), (100, 100), config)
    assert len(accepted) == info["accepted_matches"]
    np.testing.assert_array_equal(grid, info["control_match_indices"])
