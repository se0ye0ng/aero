import copy
import json
import shutil
from pathlib import Path

import pytest
import torch

from aero_ir.registration.protocol_v7 import ARCHITECTURE, LOSS_WEIGHTS, SharedVelocityMatcher
from aero_ir.registration.qualification_v4 import split_report
from aero_ir.registration.superfusion import DenseMatcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.train_antiuav300_registration_v7 import INITIAL_SHA256, epoch_selection
from scripts.verify_registration_v7 import (
    compare_arms,
    read_json,
    verify_arm,
    verify_screen,
    verify_training_trace,
)


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def trace_fixture(tmp_path):
    shards = [{"sequence_id": "a", "pairs": 40}, {"sequence_id": "b", "pairs": 40}]
    spec = {"epochs": 2, "batch_size": 16, "seed": 0, "arm": "geometry"}
    rows = []
    for epoch in range(2):
        schedule = epoch_selection(shards, epoch, 0)
        for batch in range(2):
            rows.append(
                {
                    "epoch": epoch + 1,
                    "batch": batch + 1,
                    "selection": [list(x) for x in schedule[batch * 16 : (batch + 1) * 16]],
                    "array_sha256": str(epoch + batch) * 64,
                    "loss": {"total": 0.1, "gradient_norm": 0.2},
                }
            )
    return tmp_path, spec, shards, rows


def attempt(root, spec, index, resume, rows):
    write(
        root / f"attempt_{index:03d}_runtime.json",
        {"resume_from_epoch": resume, "spec_sha256": canonical_hash(spec)},
    )
    (root / f"attempt_{index:03d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_complete_trace_has_actual_budget(trace_fixture):
    root, spec, shards, rows = trace_fixture
    attempt(root, spec, 0, 0, rows)
    result = verify_training_trace(root, spec, shards)
    assert result["verified_optimizer_steps"] == 4
    assert result["abandoned_logged_steps"] == 0


def test_resume_discards_work_after_last_saved_epoch(trace_fixture):
    root, spec, shards, rows = trace_fixture
    attempt(root, spec, 0, 0, rows[:3])
    before = verify_signature(rows)
    attempt(root, spec, 1, 1, rows[2:])
    result = verify_training_trace(root, spec, shards)
    assert result["verified_optimizer_steps"] == 4
    assert result["abandoned_logged_steps"] == 1
    assert result["consumed_data_signature"] == before


def verify_signature(rows):
    return canonical_hash(
        [{k: r[k] for k in ("epoch", "batch", "selection", "array_sha256")} for r in rows]
    )


def test_superseded_partial_line_is_not_counted_as_retained_step(trace_fixture):
    root, spec, shards, rows = trace_fixture
    attempt(root, spec, 0, 0, rows[:2])
    with (root / "attempt_000.jsonl").open("a") as f:
        f.write('{"epoch":')
    attempt(root, spec, 1, 1, rows[2:])
    assert verify_training_trace(root, spec, shards)["verified_optimizer_steps"] == 4


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "wrong_selection", "nonfinite", "missing_gradient"]
)
def test_invalid_trace_rejected(trace_fixture, defect):
    root, spec, shards, rows = trace_fixture
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[2] = copy.deepcopy(rows[1])
    elif defect == "wrong_selection":
        rows[0]["selection"][0][1] = 1000
    elif defect == "nonfinite":
        rows[0]["loss"]["total"] = float("nan")
    else:
        rows[0]["loss"].pop("gradient_norm")
    attempt(root, spec, 0, 0, rows)
    with pytest.raises(ValueError):
        verify_training_trace(root, spec, shards)


def test_unproven_resume_prefix_rejected(trace_fixture):
    root, spec, shards, rows = trace_fixture
    attempt(root, spec, 0, 1, rows[2:])
    with pytest.raises(ValueError, match="absent from previous"):
        verify_training_trace(root, spec, shards)


def screen_fixture():
    direction = {
        "finite_field": True,
        "box_in_bounds": True,
        "bbox_iou": 0.9,
        "centroid_shift_fraction": 0.0,
        "absolute_area_ratio_change": 0.0,
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
    rows = [
        {
            "sequence_id": sid,
            "ir_to_rgb_points": dict(direction),
            "rgb_to_ir_points": dict(direction),
        }
        for sid in ("a", "b")
    ]
    return {
        "rows": rows,
        "summary": split_report(rows),
        "samples": [{"selection": [[sid, 20]], "array_sha256": "a" * 64} for sid in ("a", "b")],
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "generator_training_eligible": "hold_not_qualified",
    }


@pytest.mark.parametrize("defect", ["duplicates", "fake_pass", "changed_position", "generator_go"])
def test_screen_checks_rows_not_only_aggregate(defect):
    screen = screen_fixture()
    shards = [{"sequence_id": sid, "pairs": 40} for sid in ("a", "b")]
    assert verify_screen(screen, shards)["joint_frame_pass_rate"] == 1.0
    if defect == "duplicates":
        screen["rows"][1]["sequence_id"] = "a"
    elif defect == "fake_pass":
        screen["rows"][0]["ir_to_rgb_points"]["bbox_iou"] = 0.2
    elif defect == "changed_position":
        screen["samples"][0]["selection"][0][1] = 21
    else:
        screen["generator_training_eligible"] = "pass"
    with pytest.raises(ValueError):
        verify_screen(screen, shards)


def test_missing_products_not_called_running_or_successful(tmp_path):
    result = verify_arm(tmp_path, "geometry", tmp_path / "cache")
    assert result["status"] == "pending_or_incomplete"
    assert result["process_state"] == "not_inferred"


def test_unmatched_data_cannot_be_called_mind_gain():
    arms = {}
    for arm, weight in (("geometry", 0), ("geometry_mind", 0.05)):
        arms[arm] = {
            "status": "verified_saved_artifacts_not_replayed",
            "spec": {"arm": arm, "mind_weight": weight, "seed": 0},
            "trace": {"consumed_data_signature": "same"},
            "screen_sample_signature": "same",
            "final": {"joint_frame_pass_rate": 1.0},
            "train_geometry_threshold_met": True,
        }
    assert compare_arms(arms)["matched_budget_and_data"]
    arms["geometry_mind"]["trace"]["consumed_data_signature"] = "changed"
    with pytest.raises(ValueError, match="different training"):
        compare_arms(arms)


def test_nonfinite_json_rejected(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        read_json(path)


def test_complete_saved_artifact_path_with_synthetic_fixture(tmp_path):
    """Exercise full verifier plumbing, NOT a real training/qualification experiment."""
    cache, root = tmp_path / "cache", tmp_path / "geometry"
    cache.mkdir()
    root.mkdir()
    shards = [{"sequence_id": f"sequence{i:03d}", "pairs": 16} for i in range(160)]
    manifest = {"shards": shards, "fit_split": "train", "validation_or_test_access": "none"}
    manifest["cache_manifest_sha256"] = canonical_hash(manifest)
    write(cache / "manifest.json", manifest)
    names = [
        "scripts/train_antiuav300_registration_v7.py",
        "src/aero_ir/registration/protocol_v7.py",
        "src/aero_ir/registration/shared_velocity.py",
        "src/aero_ir/registration/qualification_v4.py",
        "src/aero_ir/registration/geometry.py",
        "src/aero_ir/registration/protocol_v2.py",
    ]
    spec = {
        "architecture": ARCHITECTURE,
        "arm": "geometry",
        "epochs": 10,
        "batch_size": 8,
        "seed": 0,
        "smoke_steps": 0,
        "samples_per_sequence_per_epoch": 16,
        "fit_split": "train",
        "validation_or_test_access": "none",
        "initial_checkpoint_sha256": INITIAL_SHA256,
        "integration_steps": 7,
        "loss_weights": LOSS_WEIGHTS,
        "learning_rate": 1e-5,
        "mind_weight": 0,
        "generator_training_eligible": "hold_not_qualified",
        "cache_file_sha256": file_sha256(cache / "manifest.json"),
        "source_sha256": {name: file_sha256(name) for name in names},
    }
    write(root / "run_spec.json", spec)
    for name in names:
        dest = root / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(name), dest)
    rows = []
    for epoch in range(10):
        selection = epoch_selection(shards, epoch, 0)
        for batch in range(320):
            rows.append(
                {
                    "epoch": epoch + 1,
                    "batch": batch + 1,
                    "selection": [list(v) for v in selection[batch * 8 : (batch + 1) * 8]],
                    "array_sha256": "c" * 64,
                    "loss": {"total": 0.1, "gradient_norm": 0.2},
                }
            )
    attempt(root, spec, 0, 0, rows)
    # Deliberately synthetic counters and random weights: the verifier promises
    # saved-artifact consistency, not independent proof that SGD was executed.
    model = SharedVelocityMatcher(DenseMatcher())
    path = root / "shared_velocity_e10.pth"
    state = {
        "architecture": ARCHITECTURE,
        "shared_velocity_state": model.state_dict(),
        "spec_sha256": canonical_hash(spec),
        "completed_epochs": 10,
        "scheduler": {"last_epoch": 3200},
        "optimizer": {"state": {0: {"step": 3200}}},
    }
    torch.save(state, path)
    digest = file_sha256(path)
    report = screen_fixture()
    template = report["rows"][0]
    report["rows"] = [{**copy.deepcopy(template), "sequence_id": s["sequence_id"]} for s in shards]
    report["samples"] = [
        {"selection": [[s["sequence_id"], 8]], "array_sha256": "d" * 64} for s in shards
    ]
    report["summary"] = split_report(report["rows"])
    write(root / "initial_train_screen.json", report)
    write(
        root / "final_train_screen.json",
        {**report, "checkpoint_sha256": digest, "spec_sha256": canonical_hash(spec)},
    )
    write(
        root / "training_result.json",
        {
            "checkpoint_sha256": digest,
            "completed_epochs": 10,
            "optimizer_steps": 3200,
            "arm": "geometry",
            "generator_training_eligible": "hold_not_qualified",
        },
    )
    result = verify_arm(root, "geometry", cache)
    assert result["status"] == "verified_saved_artifacts_not_replayed"
    assert result["trace"]["verified_optimizer_steps"] == 3200
    assert result["train_geometry_threshold_met"] is True
    assert result["spec"]["generator_training_eligible"] == "hold_not_qualified"
    state["scheduler"]["last_epoch"] = 3199
    torch.save(state, path)
    with pytest.raises(ValueError, match="scheduler budget"):
        verify_arm(root, "geometry", cache)
