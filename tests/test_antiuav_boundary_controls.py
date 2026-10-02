from pathlib import Path

import numpy as np
import yaml

from scripts.probe_antiuav_boundary_controls import augmented_fit


def settings():
    return yaml.safe_load(
        Path("configs/experiment/registration_external_local_warp_cpu.yaml").read_text()
    )


def test_insufficient_matches_remain_failed_not_extrapolated():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    before, after, info = augmented_fit(x, x, np.ones(3), [(20, 20), (20, 20)], settings())
    assert before is None and after is None
    assert info["status"] == "insufficient_matches"


def test_candidate_preserves_original_controls_without_boxes():
    rng = np.random.default_rng(42)
    x = rng.uniform(20, 150, (80, 2))
    before, after, info = augmented_fit(
        x, x * 0.5 + 5, rng.uniform(size=80), [(200, 200), (100, 100)], settings()
    )
    assert before is not None and after is not None
    old = info["original_fit"]["control_match_indices"]
    assert info["expanded_match_indices"][: len(old)] == old
    np.testing.assert_allclose(after([[70, 70]]), [[40, 40]], atol=1e-8)
