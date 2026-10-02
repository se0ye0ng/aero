"""Synthetic formatting fixtures, not replacement experiment measurements."""

import json

import pytest

from scripts.export_public_results import METRICS, aggregate


def inputs():
    manifest = {
        "run_id": "synthetic_test",
        "git_sha": "a" * 40,
        "working_directory": "/private/never-export",
        "metrics": {"images": 2, "detections": 3, **dict.fromkeys(METRICS, 0.25), "per_class": {}},
    }
    proof = {
        "ok": True,
        "manifest_sha256": "b" * 64,
        "verified_from_git": ["input__source"],
        "replayed": False,
    }
    arm = {
        "trace": {"verified_optimizer_steps": 3200},
        "final": {
            "per_sequence_pass_rate": {"s0": 1, "s1": 0},
            "evaluated_pairs": 2,
            "joint_frame_pass_rate": 0.5,
        },
        "initial": {"joint_frame_pass_rate": 0.5},
        "final_checkpoint_sha256": "c" * 64,
    }
    registration = {
        "matched_budget_and_data": True,
        "arms": dict.fromkeys(("predictor_continuation", "residual_head"), arm),
        "learning_rates": {"predictor_continuation": 1e-5, "residual_head": 1e-4},
    }
    return manifest, proof, registration


def test_public_summary_separates_metrics_omits_private_paths_and_never_grants_go():
    result = aggregate(*inputs())
    assert "/private" not in json.dumps(result)
    assert result["flir_real_only"]["metrics_fraction"]["map_50_95"] == 0.25
    assert result["registration_pilots"]["arms"]["residual_head"]["joint_passes"] == 1
    assert result["three_arm_detector_comparison"]["generated_data_ap_improvement"] is None
    assert result["registration_qualification"] == "hold_not_qualified"
    assert not result["flir_real_only"]["replayed_during_this_export"]


def test_unverified_input_or_non_midpoint_panel_rejected():
    manifest, proof, registration = inputs()
    with pytest.raises(ValueError, match="unverified"):
        aggregate(manifest, {**proof, "ok": False}, registration)
    registration["arms"]["residual_head"]["final"]["per_sequence_pass_rate"]["s0"] = 0.5
    with pytest.raises(ValueError, match="one midpoint"):
        aggregate(manifest, proof, registration)
