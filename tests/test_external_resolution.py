from pathlib import Path

import numpy as np
import pytest

from scripts.probe_external_registration_landmarks import summarize
from scripts.probe_external_resolution import aggregate, reference_score, validate_replay


def test_replay_allows_continuous_roundoff_only():
    errors = np.array([0.4, 2.0, np.inf])
    perturbed = errors + 1e-10
    stats = summarize(errors, [1.0, 3.0])
    actual = summarize(perturbed, [1.0, 3.0])
    delta = validate_replay({}, actual, perturbed, {}, stats, errors, "pair")
    assert 0 < delta < 1e-8


def test_replay_roundoff_cannot_change_threshold_classification():
    expected = np.array([3.0])
    actual = expected + 1e-10
    with pytest.raises(ValueError, match="pck_all_landmarks"):
        validate_replay(
            {},
            summarize(actual, [3.0]),
            actual,
            {},
            summarize(expected, [3.0]),
            expected,
            "boundary",
        )


def test_replay_preserves_all_failure_case():
    errors = np.array([np.inf, np.inf])
    stats = summarize(errors, [3.0])
    assert validate_replay({}, stats, errors, {}, stats, errors, "unsupported") == 0


def test_replay_cannot_exchange_supported_landmark_identities():
    expected = np.array([0.5, np.inf])
    actual = expected[::-1].copy()
    # Identical aggregate scores must not hide different failed landmarks.
    with pytest.raises(ValueError, match="landmark errors"):
        validate_replay(
            {},
            summarize(actual, [3.0]),
            actual,
            {},
            summarize(expected, [3.0]),
            expected,
            "support",
        )


@pytest.mark.parametrize("change", ["fit", "pck", "count", "median", "errors", "nan"])
def test_replay_rejects_meaningful_changes(change):
    errors = np.array([0.4, 2.0, np.inf])
    stats = summarize(errors, [1.0, 3.0])
    actual = summarize(errors, [1.0, 3.0])
    fit, altered = {}, errors.copy()
    if change == "fit":
        fit = {"control_match_indices": [1]}
    elif change == "pck":
        actual["pck_all_landmarks"]["3.0"] += 1e-12
    elif change == "count":
        actual["unprojectable"] = 0
    elif change == "median":
        actual["conditional_finite_median"] += 1e-5
    elif change == "errors":
        altered[0] += 1e-5
    else:
        altered[0] = np.nan
    with pytest.raises(ValueError, match="pair"):
        validate_replay(fit, actual, altered, {}, stats, errors, "pair")


def test_failure_landmarks_remain_in_aggregate():
    rows = [
        dict(
            long_side=size,
            summary=summarize(np.array([0.0, 1.0, np.inf, np.inf]), [1.0, 3.0, 5.0, 10.0]),
        )
        for size in (640, 1280)
        for _ in range(4)
    ]
    result = aggregate(rows)
    assert len(result) == 2
    assert all(r["landmarks"] == 16 and r["pairs"] == 4 for r in result)
    assert all(r["macro_pck"]["3.0"] == 0.5 for r in result)


def test_reference_score_no_warp_is_all_failure(monkeypatch):
    points = np.array([[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]])
    monkeypatch.setattr(np, "loadtxt", lambda path: points.copy())
    errors, summary = reference_score(
        Path("/unused"), "pair", (40, 40), (40, 40), None, [1.0, 3.0, 5.0, 10.0]
    )
    assert np.isinf(errors).all()
    assert summary["landmarks"] == summary["unprojectable"] == 4
    assert summary["pck_all_landmarks"]["3.0"] == 0.0


def test_reference_queries_do_not_change_fitted_mapping(monkeypatch):
    points = np.array([[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]])
    monkeypatch.setattr(np, "loadtxt", lambda path: points.copy())
    errors, summary = reference_score(
        Path("/unused"),
        "pair",
        (40, 40),
        (40, 40),
        lambda xy: xy + [2.0, 0.0],
        [1.0, 3.0, 5.0, 10.0],
    )
    np.testing.assert_array_equal(errors, [2.0] * 4)
    assert summary["pck_all_landmarks"]["1.0"] == 0.0
    assert summary["pck_all_landmarks"]["3.0"] == 1.0
