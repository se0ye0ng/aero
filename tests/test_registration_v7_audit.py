"""Analytic/synthetic audit fixtures only; no trained or qualified result claims."""

import copy
import json
import sys

import numpy as np
import pytest
import torch

from aero_ir.registration.qualification_v4 import split_report
from aero_ir.utils.manifest import file_sha256
from scripts import audit_antiuav300_registration_v7 as audit
from scripts.audit_antiuav300_dense_registration import FramePair


class CentreTranslation:
    def __init__(self, dx=0):
        self.dx = dx
        self.calls = 0

    def fields(self, ir, rgb):
        self.calls += 1
        field = ir.new_zeros(len(ir), 2, *ir.shape[-2:])
        field[:, 0] = self.dx
        return field, -field

    def __call__(self, *args, **kwargs):
        raise AssertionError("must not use the raw legacy field interface")


def pair(split="train", frame=0, dx=0):
    return FramePair(
        split=split,
        sequence_id=split + "_sequence",
        frame_index=frame,
        visible=np.full((32, 64, 3), 51, np.uint8),
        infrared=np.full((32, 64, 3), 204, np.uint8),
        source_box=np.asarray([0.5 + dx / 2, 0.5, 0.25, 0.25], np.float32),
        target_box=np.asarray([0.5, 0.5, 0.25, 0.25], np.float32),
    )


@pytest.mark.parametrize("shift", [0, 2 / 64, -2 / 64])
def test_direct_centre_fields_and_correct_directional_box_roles(shift):
    model = CentreTranslation(shift)
    rows = audit.infer_rows(model, [pair(dx=shift), pair(frame=1, dx=shift)], "cpu")
    assert model.calls == 1
    for row in rows:
        for key in ("ir_to_rgb_points", "rgb_to_ir_points"):
            assert row[key]["bbox_iou"] > 0.9999
            assert row[key]["roi_cycle_p95_pixels"] < 1e-5
            assert row[key]["cycle_p95_pixels"] < 1e-5
            assert audit.direction_pass(row[key])
        assert len(row["array_sha256"]) == 64


def test_pair_hash_binds_pixels_boxes_and_frame_identity():
    original = pair()
    for key in ("visible", "infrared", "source_box", "target_box"):
        changed = copy.deepcopy(original)
        getattr(changed, key).flat[0] += 1
        assert audit.pair_digest(changed) != audit.pair_digest(original)
    assert audit.pair_digest(pair(frame=1)) != audit.pair_digest(original)
    assert audit.pair_digest(pair(split="val")) != audit.pair_digest(original)


def test_nonfinite_fields_are_counted_as_failures_not_dropped():
    class BrokenModel:
        def fields(self, ir, rgb):
            field = ir.new_full((len(ir), 2, *ir.shape[-2:]), float("nan"))
            return field, field

    row = audit.infer_rows(BrokenModel(), [pair()], "cpu")[0]
    json.dumps(row, allow_nan=False)
    counter = audit.Coverage({"train": {("train_sequence", 0)}})
    counter.add(row)
    metrics = counter.finish()["train"]
    assert metrics["evaluated_pairs"] == 1
    assert metrics["joint_frame_pass_rate"] == 0


def test_streaming_summary_equals_v4_and_requires_same_frame_bidirectionality():
    pairs = [pair(), pair(frame=1), pair(split="val")]
    rows = audit.infer_rows(CentreTranslation(), pairs, "cpu")
    rows[0]["ir_to_rgb_points"]["bbox_iou"] = 0.1
    rows[1]["rgb_to_ir_points"]["bbox_iou"] = 0.1
    expected = {
        s: {(p.sequence_id, p.frame_index) for p in pairs if p.split == s} for s in audit.SPLITS
    }
    counter = audit.Coverage(expected)
    for row in rows:
        counter.add(row)
    metrics = counter.finish()
    assert metrics == {s: split_report([r for r in rows if r["split"] == s]) for s in audit.SPLITS}
    assert metrics["train"]["joint_frame_pass_rate"] == 0
    assert metrics["val"]["joint_frame_pass_rate"] == 1


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unexpected", "test"])
def test_incomplete_or_wrong_observations_rejected(defect):
    expected = {"train": {("train_sequence", 0), ("train_sequence", 1)}}
    counter = audit.Coverage(expected)
    row = audit.infer_rows(CentreTranslation(), [pair()], "cpu")[0]
    counter.add(row)
    with pytest.raises(ValueError):
        if defect == "missing":
            counter.finish()
        elif defect == "duplicate":
            counter.add(row)
        elif defect == "unexpected":
            counter.add({**row, "frame_index": 999})
        else:
            counter.add({**row, "split": "test"})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def panel_fixture(tmp_path):
    root, cache = tmp_path / "dataset", tmp_path / "cache"
    for split in audit.SPLITS:
        name = split + "_sequence"
        write(root / "label_new" / f"{split}.json", {name: ["day"]})
        for modality in ("visible", "infrared"):
            write(
                root / split / name / f"{modality}.json",
                {"exist": [1] * 12, "gt_rect": [[10, 10, 30, 20]] * 12},
            )
    write(
        cache / "manifest.json",
        {
            "fit_split": "train",
            "validation_or_test_access": "none",
            "split_manifest_sha256": file_sha256(root / "label_new/train.json"),
            "shards": [{"sequence_id": "train_sequence", "pairs": 12}],
        },
    )
    return root, cache


@pytest.mark.parametrize("samples,count", [(8, 8), (0, 12)])
def test_panel_frozen_selection_without_test_access(panel_fixture, samples, count):
    root, cache = panel_fixture
    expected, panel = audit.build_panel(root, cache, samples)
    assert {s: len(v) for s, v in expected.items()} == {"train": count, "val": count}
    assert panel["official_sequences"] == {"train": 1, "val": 1}
    assert panel["sequences_without_eligible_pairs"] == {"train": [], "val": []}
    assert panel["test_access"] == "none"
    assert not (root / "test").exists()
    assert audit.build_panel(root, cache, samples) == (expected, panel)


def test_panel_rejects_overlap_and_changed_train_manifest(panel_fixture):
    root, cache = panel_fixture
    write(root / "label_new/val.json", {"train_sequence": []})
    with pytest.raises(ValueError, match="overlap"):
        audit.build_panel(root, cache, 8)
    write(root / "label_new/val.json", {"val_sequence": []})
    write(root / "label_new/train.json", {"train_sequence": ["different metadata"]})
    with pytest.raises(ValueError, match="differs from training cache"):
        audit.build_panel(root, cache, 8)


def test_zero_eligible_sequence_remains_explicit(panel_fixture):
    root, cache = panel_fixture
    write(
        root / "val/val_sequence/visible.json", {"exist": [0] * 12, "gt_rect": [[0, 0, 0, 0]] * 12}
    )
    expected, panel = audit.build_panel(root, cache, 8)
    assert expected["val"] == set()
    assert panel["sequences_without_eligible_pairs"]["val"] == ["val_sequence"]


def test_preflight_missing_pilots_never_touches_cuda_or_frames(
    panel_fixture, tmp_path, monkeypatch
):
    root, cache = panel_fixture
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--root",
            str(root),
            "--cache-root",
            str(cache),
            "--run-root",
            str(tmp_path / "no_pilots"),
            "--out-dir",
            str(output),
            "--preflight-only",
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not use CUDA or decode video")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(audit, "_iter_pairs", forbidden)
    assert audit.main() == 2
    report = json.loads((output / "preflight.json").read_text())
    assert report["ready_for_engineering_inference"] is False
    assert report["pixel_inference_executed"] is False
    assert report["generator_training_eligible"] == "hold_not_qualified"
    assert not (output / "report.json").exists()
    with pytest.raises(SystemExit):
        audit.main()  # Never overwrite even a pending preflight.


def test_complete_mock_inference_writes_rows_without_granting_qualification(
    panel_fixture, tmp_path, monkeypatch
):
    root, cache = panel_fixture
    run, out = tmp_path / "pilots", tmp_path / "audit"
    for arm in audit.ARMS:
        (run / arm).mkdir(parents=True)
        (run / arm / "shared_velocity_e10.pth").write_bytes(b"test loader mocked, not a real pilot")
    # Model-completion validation itself is covered by test_registration_v7_verifier.
    monkeypatch.setattr(
        audit,
        "verify_arm",
        lambda folder, arm, cache: {
            "artifacts_sha256": {
                "shared_velocity_e10.pth": file_sha256(folder / "shared_velocity_e10.pth")
            }
        },
    )
    monkeypatch.setattr(
        audit,
        "compare_arms",
        lambda arms: {"candidates_for_heldout_engineering_screen": ["geometry"]},
    )
    monkeypatch.setattr(audit, "load_trained", lambda path, device: (CentreTranslation(), {}))
    selected, _ = audit.build_panel(root, cache, 8)
    pairs = [pair(s, index) for s in audit.SPLITS for _, index in sorted(selected[s])]
    monkeypatch.setattr(audit, "_iter_pairs", lambda *args: iter(pairs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--root",
            str(root),
            "--cache-root",
            str(cache),
            "--run-root",
            str(run),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
        ],
    )
    assert audit.main() == 0
    report = json.loads((out / "report.json").read_text())
    assert len(set(report["consumed_pair_signatures"].values())) == 1
    assert report["exact_eligible_pair_coverage"] is True
    for arm in audit.ARMS:
        assert len((out / f"{arm}.jsonl").read_text().splitlines()) == 16
        assert report["metrics"][arm]["val"]["joint_frame_pass_rate"] == 1
        assert report["gates"][arm]["geometric_screen"] == "pass"
        assert report["gates"][arm]["exhaustive_geometry"] == "hold"
        assert report["gates"][arm]["generator_training_eligible"] == "hold"
        assert report["gates"][arm]["independent_correspondence"] == "hold_not_evaluated"
