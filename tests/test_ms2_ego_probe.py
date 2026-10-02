import numpy as np

from scripts.probe_ms2_ego_motion import measure


def test_joint_denominator_keeps_unassociated_sources():
    source = np.full((1, 3), 256, dtype=np.uint16)
    target = np.array([[256, 0, 0]], dtype=np.uint16)
    result, associated, errors = measure(source, target, np.eye(3), np.eye(3), np.eye(4))
    assert result["source_points"] == 3
    assert result["associated_points"] == 2
    assert result["joint_counts"]["0.01"] == 2
    assert associated.tolist() == [True, True, False]
    assert np.isnan(errors[2])


def test_bad_depth_does_not_become_joint_success():
    source = np.full((2, 2), 256, dtype=np.uint16)
    target = np.full((2, 2), 512, dtype=np.uint16)
    result, _, _ = measure(source, target, np.eye(3), np.eye(3), np.eye(4))
    assert result["associated_points"] == 4
    assert set(result["joint_counts"].values()) == {0}


def test_missing_depth_yields_no_association():
    source = np.full((2, 2), 256, dtype=np.uint16)
    result, _, _ = measure(source, source * 0, np.eye(3), np.eye(3), np.eye(4))
    assert result["source_points"] == 4
    assert result["associated_points"] == 0
    assert result["conditional_abs_depth_m_p50_p90"] is None
