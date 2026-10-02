import numpy as np

from scripts.analyze_external_support_loss import support_masks


def test_distinguishes_control_loss_from_raw_support_absence():
    raw = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    controls = np.array([[2, 2], [6, 2], [2, 6]], dtype=float)
    result = support_masks(raw, controls, [[3, 3], [8, 8], [20, 20]])
    np.testing.assert_array_equal(result["raw_hull"], [True, True, False])
    np.testing.assert_array_equal(result["control_hull"], [True, False, False])
    assert np.isfinite(result["nearest_raw_rgb_px"]).all()
