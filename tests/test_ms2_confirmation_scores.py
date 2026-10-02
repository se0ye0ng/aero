import numpy as np
import pytest

from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_confirmation import add_identity, candidate_scores


def test_failed_candidate_keeps_fixed_denominator():
    source = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    depth = np.array([10.0, 10.0, np.nan])
    moved = np.eye(4)
    moved[0, 3] = 10000.0
    scores, eligible = candidate_scores(
        source, source, depth, np.eye(3), np.eye(3), np.eye(4), dict(author=np.eye(4), bad=moved)
    )
    np.testing.assert_array_equal(eligible, [True, True, False])
    assert scores["author"]["all_matches"]["points"] == 3
    assert scores["author"]["all_matches"]["fraction_all"]["3.0"] == 2 / 3
    assert scores["bad"]["fixed_supported"]["points"] == 2
    assert scores["bad"]["fixed_supported"]["supported"] == 0
    assert scores["bad"]["fixed_supported"]["fraction_all"]["3.0"] == 0.0


def test_empty_case_is_recorded_not_dropped():
    empty = np.empty((0, 2))
    scores, eligible = candidate_scores(
        empty, empty, np.empty(0), np.eye(3), np.eye(3), np.eye(4), dict(author=np.eye(4))
    )
    assert len(eligible) == 0
    assert scores["author"]["fixed_supported"]["points"] == 0
    assert scores["author"]["fixed_supported"]["fraction_all"]["3.0"] is None


def test_adding_dependencies_cannot_overwrite_frozen_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("frozen: true")
    known = {str(path): file_sha256(path)}
    original = known.copy()
    add_identity(known, path)
    assert known == original
    path.write_text("frozen: false")
    with pytest.raises(ValueError, match="frozen candidate"):
        add_identity(known, path)
    assert known == original
