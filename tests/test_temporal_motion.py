import numpy as np
import pytest

from scripts.probe_antiuav_temporal_motion import residual_motion


def test_camera_translation_removed():
    f = np.ones((2, 20, 20, 2)) * [3, -2]
    score, mask, _ = residual_motion(f)
    assert not mask.any()
    assert np.max(score) == 0


def test_two_sided_relative_motion_detected():
    f = np.zeros((2, 20, 20, 2))
    f[0, 5:8, 5:8, 0] = 2
    f[1, 5:8, 5:8, 0] = -2
    _, mask, _ = residual_motion(f)
    assert mask.sum() == 9
    f[1] = 0
    assert not residual_motion(f)[1].any()


def test_invalid_flow_rejected():
    with pytest.raises(ValueError):
        residual_motion(np.full((2, 4, 4, 2), np.nan))
