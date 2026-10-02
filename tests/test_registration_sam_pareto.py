import json

import numpy as np
import pytest

from aero_ir.registration.silhouette_pareto import (
    IDENTITY,
    candidates_around,
    compare,
    non_deteriorating,
    select,
)
from scripts.probe_registration_sam_pareto import load_inputs


def test_combined_improvement_is_not_two_metric_improvement():
    baseline = {"dice_loss": 0.7, "chamfer_model_px": 7.0}
    bad = {"dice_loss": 0.3, "chamfer_model_px": 17.0}
    records = [
        {"parameters": IDENTITY, "score": baseline},
        {"parameters": (1.1, 0, 2, 0), "score": bad},
    ]
    assert select(records, constrained=False)["score"] == bad
    assert select(records, constrained=True)["parameters"] == IDENTITY


def test_missing_and_nonfinite_baselines_fail_closed():
    valid = {"dice_loss": 0.2, "chamfer_model_px": 1.0}
    assert not non_deteriorating(valid, None)
    assert not non_deteriorating({**valid, "dice_loss": float("nan")}, valid)
    assert select([{"parameters": IDENTITY, "score": None}], constrained=True) is None


def test_candidate_pool_deterministic_and_includes_reference():
    p = {"scale": 1.0, "angle_degrees": 0.0, "dx": 16.0, "dy": 0.0}
    candidates = candidates_around(p)
    assert len(candidates) == 82
    assert candidates == candidates_around(p)
    assert IDENTITY in candidates and (1.0, 0.0, 16.0, 0.0) in candidates


def test_known_shift_retained_with_inverse():
    a = np.zeros((256, 256), bool)
    a[90:110, 90:140] = True
    a[110:150, 90:105] = True
    b = np.roll(a, 16, axis=1)

    def data(mask):
        return {
            "masks": np.stack([mask] * 3),
            "excluded": np.zeros_like(mask),
            "meta": {"native_to_crop": np.eye(3), "crop_to_native": np.eye(3)},
        }

    old = {"parameters": {"scale": 1.0, "angle_degrees": 0.0, "dx": 16.0, "dy": 0.0}}
    result = compare(data(a), data(b), old)["non_deteriorating"]
    assert result["status"] == "fit_pseudo_masks_only"
    assert result["score"] == {"dice_loss": 0.0, "chamfer_model_px": 0.0}
    assert np.allclose(np.array(result["matrix"]) @ result["inverse_matrix"], np.eye(3))
    assert all(r["selected"]["dice_loss"] == 0 for r in result["perturbed_prompt_checks_not_gt"])


@pytest.mark.parametrize("mode", ["replay", "unknown"])
def test_refuses_replayed_or_unknown_mask_origin(tmp_path, mode):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"experiment": "sam_centred_roi_similarity_v2", "mode": mode}))
    with pytest.raises(ValueError, match="fresh train-only"):
        load_inputs(path)


def test_refuses_missing_or_changed_hashed_artifact(tmp_path):
    path = tmp_path / "report.json"
    (tmp_path / "mask.npz").write_bytes(b"tampered")
    path.write_text(
        json.dumps(
            {
                "experiment": "sam_centred_roi_similarity_v2",
                "mode": "extract",
                "evaluated_split": "train",
                "validation_or_test_access": "none",
                "artifacts_sha256": {"mask.npz": "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="invalid source artifact"):
        load_inputs(path)


def test_refuses_artifact_path_escape(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "experiment": "sam_centred_roi_similarity_v2",
                "mode": "extract",
                "evaluated_split": "train",
                "validation_or_test_access": "none",
                "artifacts_sha256": {"../outside.npz": "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="invalid source artifact"):
        load_inputs(path)
