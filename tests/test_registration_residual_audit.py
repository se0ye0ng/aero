"""Synthetic audit fixtures, not completed-pilot or correspondence evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aero_ir.utils.manifest import file_sha256
from scripts import audit_registration_residual_pilot as audit
from scripts.audit_antiuav300_dense_registration import FramePair


def comparison(rate=0.95, macro=0.95):
    return {
        "arms": {
            a: {"final": {"joint_frame_pass_rate": rate, "sequence_macro_pass_rate": macro}}
            for a in audit.ARMS
        }
    }


@pytest.mark.parametrize(
    "rate,macro,ready",
    [(0.95, 0.95, True), (0.949, 1, False), (1, 0.949, False), (float("nan"), 1, False)],
)
def test_both_frozen_train_rates_required(rate, macro, ready):
    assert bool(audit.candidates(comparison(rate, macro))) == ready


class IdentityCentreModel:
    def fields(self, ir, rgb):
        value = ir.new_zeros(len(ir), 2, *ir.shape[-2:])
        return value, value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    args = SimpleNamespace(
        root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        run_root=tmp_path / "runs",
        out_dir=tmp_path / "audit",
        device="cpu",
        batch_size=2,
        samples_per_sequence=8,
        preflight_only=False,
    )
    for split in audit.SPLITS:
        sequence = split + "_sequence"
        write(args.root / "label_new" / f"{split}.json", {sequence: ["day"]})
        for modality in ("visible", "infrared"):
            write(
                args.root / split / sequence / f"{modality}.json",
                {"exist": [1] * 8, "gt_rect": [[10, 10, 30, 20]] * 8},
            )
    write(
        args.cache_root / "manifest.json",
        {
            "fit_split": "train",
            "validation_or_test_access": "none",
            "split_manifest_sha256": file_sha256(args.root / "label_new/train.json"),
            "shards": [{"sequence_id": "train_sequence", "pairs": 8}],
        },
    )
    for arm in audit.ARMS:
        for name in (
            "run_spec.json",
            "training_result.json",
            "final.pth",
            "initial_train_screen.json",
            "final_train_screen.json",
        ):
            write(args.run_root / arm / name, {"test_double": True})
    # Integrity verification is deliberately mocked; the real comparison's own
    # tests cover rejecting incomplete/smoke products. These tests cover wiring.
    monkeypatch.setattr(audit, "compare", lambda *args: comparison())
    monkeypatch.setattr(
        audit,
        "load_checkpoint",
        lambda path, device: (IdentityCentreModel(), {"arm": path.parent.name}),
    )
    pairs = [
        FramePair(
            split=split,
            sequence_id=split + "_sequence",
            frame_index=i,
            visible=np.full((32, 32, 3), 80, np.uint8),
            infrared=np.full((32, 32, 3), 160, np.uint8),
            source_box=np.asarray([0.5, 0.5, 0.25, 0.25], np.float32),
            target_box=np.asarray([0.5, 0.5, 0.25, 0.25], np.float32),
        )
        for split in audit.SPLITS
        for i in range(8)
    ]
    monkeypatch.setattr(audit, "_iter_pairs", lambda *args: iter(pairs))
    return args, pairs


@pytest.mark.parametrize("reason", ["missing", "failed_screen", "invalid_evidence"])
def test_blocked_runs_never_access_panel_pixels_or_cuda(fixture, monkeypatch, reason):
    args, _ = fixture
    if reason == "missing":
        args.run_root = args.run_root / "absent"
    elif reason == "failed_screen":
        monkeypatch.setattr(audit, "compare", lambda *args: comparison(0.8, 0.8))
    else:

        def bad_evidence(*args):
            raise ValueError("source drift")

        monkeypatch.setattr(audit, "compare", bad_evidence)

    def forbidden(*args, **kwargs):
        raise AssertionError("blocked audit accessed evaluation inputs or GPU")

    monkeypatch.setattr(audit, "build_panel", forbidden)
    monkeypatch.setattr(audit, "_iter_pairs", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert audit.run(args) == 2
    plan = json.loads((args.out_dir / "preflight.json").read_text())
    assert not plan["ready_for_engineering_inference"]
    assert not plan["pixel_inference_executed"]
    assert not (args.out_dir / "panel.json").exists()
    assert not (args.out_dir / "report.json").exists()


def test_ready_preflight_never_decodes_frames_or_initializes_cuda(fixture, monkeypatch):
    args, _ = fixture
    args.preflight_only = True

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight used pixels/GPU")

    monkeypatch.setattr(audit, "_iter_pairs", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert audit.run(args) == 0
    plan = json.loads((args.out_dir / "preflight.json").read_text())
    assert plan["selected_pairs"] == {"train": 8, "val": 8}
    assert plan["generator_training_eligible"] == "hold_not_qualified"
    assert not (args.out_dir / "report.json").exists()


def test_mock_inference_preserves_coordinates_paired_counts_and_hold(fixture):
    args, _ = fixture
    assert audit.run(args) == 0
    report = json.loads((args.out_dir / "report.json").read_text())
    assert report["generator_training_eligible"] == "hold_not_qualified"
    assert report["exact_eligible_pair_coverage"]
    assert len(set(report["consumed_pair_signatures"].values())) == 1
    for arm in audit.ARMS:
        assert report["metrics"][arm]["val"]["joint_frame_pass_rate"] == 1
        assert report["gates"][arm]["generator_training_eligible"] != "pass"
    assert report["paired_outcomes"]["val"]["both_pass"] == 8
    with pytest.raises(FileExistsError):
        audit.run(args)


def test_mid_inference_checkpoint_drift_never_writes_report(fixture, monkeypatch):
    args, pairs = fixture

    def changed(*unused):
        yield from pairs
        (args.run_root / audit.ARMS[0] / "final.pth").write_bytes(b"changed")

    monkeypatch.setattr(audit, "_iter_pairs", changed)
    with pytest.raises(ValueError, match="pilot artifact drift"):
        audit.run(args)
    assert not (args.out_dir / "report.json").exists()


def test_cross_arm_frame_mismatch_rejected(fixture):
    _, pairs = fixture
    row = audit.infer_rows(IdentityCentreModel(), pairs[:1], "cpu")[0]
    other = copy.deepcopy(row)
    other["frame_index"] += 1
    with pytest.raises(ValueError, match="arm observations differ"):
        audit.paired_key(row, other)


def test_wrong_architecture_arm_rejected_before_reading_frames(fixture, monkeypatch):
    args, _ = fixture
    monkeypatch.setattr(
        audit, "load_checkpoint", lambda *args: (IdentityCentreModel(), {"arm": "geometry_mind"})
    )
    with pytest.raises(ValueError, match="checkpoint arm differs"):
        audit.run(args)
    assert not (args.out_dir / "report.json").exists()
