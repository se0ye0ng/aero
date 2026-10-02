"""Synthetic evidence consistency checks, not registration qualification."""

import copy
import json
import sys

import pytest

from aero_ir.registration.geometry import CONVENTION
from aero_ir.registration.qualification_v4 import THRESHOLDS, gate_report, split_report
from aero_ir.utils.manifest import canonical_hash
from scripts import prepare_generator_training as planner


def sign(value, key):
    value[key] = canonical_hash({k: v for k, v in value.items() if k != key})
    return value


@pytest.fixture
def evidence():
    good = {
        "finite_field": True,
        "box_in_bounds": True,
        "bbox_iou": 0.9,
        "centroid_shift_fraction": 0.01,
        "absolute_area_ratio_change": 0.01,
        "valid_fraction": 1.0,
        "cycle_valid_fraction": 1.0,
        "positive_jacobian_fraction": 1.0,
        "roi_valid_fraction": 1.0,
        "roi_positive_jacobian_fraction": 1.0,
        "cycle_p95_pixels": 0.0,
        "roi_cycle_p95_pixels": 0.0,
        "roi_cycle_max_pixels": 0.0,
        "roi_diagonal_pixels": 20.0,
    }
    rows = {
        s: [
            {
                "sequence_id": s + "_seq",
                "frame_index": 0,
                "ir_to_rgb_points": copy.deepcopy(good),
                "rgb_to_ir_points": copy.deepcopy(good),
            }
        ]
        for s in ("train", "val")
    }
    metrics = {s: split_report(items) for s, items in rows.items()}
    report = {
        "kind": "antiuav300_coordinate_consistent_geometry_audit_v4",
        "schema_version": 4,
        "coordinate_convention": CONVENTION,
        "thresholds": copy.deepcopy(THRESHOLDS),
        "rows": rows,
        "metrics": metrics,
        "annotation_sha256": {"label_new/train.json": "a" * 64},
        "data_usage": {
            "splits": ["train", "val"],
            "test_access": "none",
            "exact_pair_coverage": True,
            "samples_per_sequence": 0,
        },
        "gates": gate_report(metrics, exhaustive=True, complete_coverage=True),
    }
    cache = {
        "kind": "antiuav300_registration_v2_full_train_cache",
        "schema_version": 2,
        "fit_split": "train",
        "validation_or_test_access": "none",
        "split_manifest_sha256": "a" * 64,
        "sequences": 1,
        "pairs": 1,
        "shards": [{"sequence_id": "train_seq", "pairs": 1, "shard_sha256": "b" * 64}],
    }
    return sign(report, "audit_sha256"), sign(cache, "cache_manifest_sha256")


def test_perfect_engineering_geometry_is_not_physical_authorization(evidence):
    result = planner.validate_evidence(*evidence)
    assert result["gates"]["exhaustive_geometry"] == "pass"
    assert result["registration_gate"] == "hold"
    assert result["scientific_authorization_verified"] is False
    assert result["registered_conditioning_export_verified"] is False


def test_pass_string_with_recomputed_hash_still_rejected(evidence):
    report, cache = evidence
    report["gates"]["generator_training_eligible"] = "pass"
    sign(report, "audit_sha256")
    with pytest.raises(ValueError, match="no manual pass override"):
        planner.validate_evidence(report, cache)


def test_unknown_protocol_cannot_authorize(evidence):
    with pytest.raises(ValueError, match="unsupported qualification"):
        planner.validate_evidence({"gates": {"generator_training_eligible": "pass"}}, evidence[1])


@pytest.mark.parametrize("part", ["thresholds", "metrics", "rows"])
def test_rehashed_contradictory_evidence_is_rejected(evidence, part):
    report, cache = evidence
    if part == "thresholds":
        report[part]["minimum_bbox_iou"] = 0.1
    elif part == "metrics":
        report[part]["train"]["joint_frame_pass_rate"] = 0.0
    else:
        report[part]["train"].append(copy.deepcopy(report[part]["train"][0]))
    sign(report, "audit_sha256")
    with pytest.raises(ValueError):
        planner.validate_evidence(report, cache)


@pytest.mark.parametrize(
    "key,value",
    [("fit_split", None), ("validation_or_test_access", None), ("split_manifest_sha256", "c" * 64)],
)
def test_cache_requires_explicit_train_identity(evidence, key, value):
    report, cache = evidence
    cache[key] = value
    sign(cache, "cache_manifest_sha256")
    with pytest.raises(ValueError):
        planner.validate_evidence(report, cache)


def test_drift_without_rehash_rejected(evidence):
    report, cache = evidence
    cache["pairs"] = 12
    with pytest.raises(ValueError, match="cache content hash"):
        planner.validate_evidence(report, cache)


def cli_args(tmp_path, evidence):
    report, cache = (tmp_path / name for name in ("report.json", "cache.json"))
    report.write_text(json.dumps(evidence[0]))
    cache.write_text(json.dumps(evidence[1]))
    return [
        "prepare",
        "--qualification-report",
        str(report),
        "--pair-manifest",
        str(cache),
        "--output-dir",
        str(tmp_path / "plan"),
        "--generator",
        "synthetic_baseline",
        "--real-count",
        "0",
        "--generated-count",
        "0",
        "--command",
        "never-run",
    ]


def test_blocked_plan_does_not_launch_or_overwrite(tmp_path, evidence, monkeypatch):
    args = cli_args(tmp_path, evidence) + ["--plan-only"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(planner.subprocess, "run", lambda *a, **k: pytest.fail("command launched"))
    assert planner.main() == 0
    path = tmp_path / "plan" / "generator_training_plan.json"
    content = path.read_bytes()
    assert json.loads(content)["status"] == "blocked_by_unqualified_evidence"
    with pytest.raises(FileExistsError):
        planner.main()
    assert path.read_bytes() == content


def test_execution_mode_blocks_without_launch(tmp_path, evidence, monkeypatch):
    monkeypatch.setattr(sys, "argv", cli_args(tmp_path, evidence))
    monkeypatch.setattr(planner.subprocess, "run", lambda *a, **k: pytest.fail("command launched"))
    with pytest.raises(SystemExit, match="qualification is not passed"):
        planner.main()
