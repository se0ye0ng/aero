import numpy as np

from scripts.probe_registration_minima import save_matches, score_local
from tests.test_external_local_warp import config


def test_local_empty_predictions_are_failures_not_identity():
    points, confidence = np.empty((0, 2)), np.empty(0)
    arrays = [points, points, confidence] * 2
    result = score_local(
        arrays, [np.array([10, 10, 20, 20])] * 2, dict(config(), maximum_controls=144), "grid"
    )
    assert result["joint_box_proxy_pass"] is False
    assert all(v["fit_status"] == "insufficient_matches" for v in result["directions"].values())


def test_known_bidirectional_shift_and_npz_output(tmp_path):
    xx, yy = np.meshgrid(np.linspace(10, 90, 9), np.linspace(10, 90, 9))
    p = np.column_stack((xx.ravel(), yy.ravel()))
    q, c = p + [4, 8], np.ones(len(p))
    arrays = [p, q, c, q, p, c]
    box = np.array([20, 20, 60, 60])
    result = score_local(
        arrays, [box, box + [4, 8, 4, 8]], dict(config(), maximum_controls=144), "fps"
    )
    assert result["joint_box_proxy_pass"] is True
    identity = save_matches(tmp_path / "matches.npz", arrays)
    assert identity["matches_file"] == "matches.npz"
    assert len(identity["matches_sha256"]) == 64
    with np.load(tmp_path / "matches.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["points0"], p)
