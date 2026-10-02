"""Synthetic audit wiring tests; not completed300-epoch or physical-GT evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aero_ir.utils.manifest import file_sha256
from scripts import audit_registration_replay as audit
from scripts.audit_antiuav300_dense_registration import FramePair


def comparison(rate=0.95, macro=0.95):
    return {
        "arms": {
            arm: {
                "panels": {"2": {"joint_frame_pass_rate": rate, "sequence_macro_pass_rate": macro}}
            }
            for arm in audit.ARMS
        }
    }


@pytest.mark.parametrize(
    "rate,macro,ready",
    [
        (0.95, 0.95, True),
        (0.949, 1, False),
        (1, 0.949, False),
        (float("nan"), 1, False),
        (float("inf"), 1, False),
        (True, 1, False),
    ],
)
def test_unchanged_joint_and_sequence_thresholds(rate, macro, ready):
    assert bool(audit.candidates(comparison(rate, macro))) == ready


class IdentityModel:
    def fields(self, ir, visible):
        field = ir.new_zeros(len(ir), 2, *ir.shape[-2:])
        return field, field


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    args = SimpleNamespace(
        root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        base_root=tmp_path / "base",
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
            "final_train_div2.json",
            "final_train_div4.json",
        ):
            write(args.run_root / arm / name, {"test_double": True, "arm": arm})
    # Mock only training-product integrity/model loading here. Training verifier
    # tests separately cover incomplete/smoke/mismatched checkpoint rejection.
    monkeypatch.setattr(audit, "compare", lambda *args: comparison())
    monkeypatch.setattr(audit, "load_checkpoint", lambda *args: (IdentityModel(), {}))
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
def test_blocked_audit_never_reads_validation_or_cuda(fixture, monkeypatch, reason):
    args, _ = fixture
    if reason == "missing":
        args.run_root = args.run_root / "absent"
    elif reason == "failed_screen":
        monkeypatch.setattr(audit, "compare", lambda *args: comparison(0.8, 0.8))
    else:

        def invalid(*args):
            raise ValueError("checkpoint drift")

        monkeypatch.setattr(audit, "compare", invalid)

    def forbidden(*args, **kwargs):
        raise AssertionError("blocked audit touched validation inputs or GPU")

    monkeypatch.setattr(audit, "build_panel", forbidden)
    monkeypatch.setattr(audit, "_iter_pairs", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert audit.run(args) == 2
    plan = json.loads((args.out_dir / "preflight.json").read_text())
    assert not plan["ready_for_engineering_inference"]
    assert plan["training_process_state"] == "not_inferred_from_files"
    assert not (args.out_dir / "panel.json").exists()
    assert not (args.out_dir / "report.json").exists()


def test_ready_preflight_does_not_decode_or_initialize_cuda(fixture, monkeypatch):
    args, _ = fixture
    args.preflight_only = True

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight used pixels/GPU")

    monkeypatch.setattr(audit, "_iter_pairs", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert audit.run(args) == 0
    plan = json.loads((args.out_dir / "preflight.json").read_text())
    assert plan["selected_pairs"] == {"train": 8, "val": 8}
    assert plan["generator_training_eligible"] == audit.HOLD
    for name, digest in plan["source_sha256"].items():
        assert file_sha256(args.out_dir / "sources" / name) == digest


@pytest.mark.parametrize("samples", [0, 8])
def test_mock_inference_coverage_pairing_and_hold(fixture, samples):
    args, _ = fixture
    args.samples_per_sequence = samples
    assert audit.run(args) == 0
    report = json.loads((args.out_dir / "report.json").read_text())
    assert report["generator_training_eligible"] == audit.HOLD
    assert report["exact_eligible_pair_coverage"]
    assert len(set(report["consumed_pair_signatures"].values())) == 1
    for arm in audit.ARMS:
        assert report["metrics"][arm]["val"]["joint_frame_pass_rate"] == 1
        assert report["gates"][arm]["generator_training_eligible"] != "pass"
    assert report["paired_outcomes"]["val"]["both_pass"] == 8
    with pytest.raises(FileExistsError):
        audit.run(args)


@pytest.mark.parametrize("fault", ["drift", "missing_frame", "duplicate_frame"])
def test_faults_never_publish_completion_report(fixture, monkeypatch, fault):
    args, pairs = fixture

    def changed(*unused):
        if fault == "missing_frame":
            yield from pairs[:-1]
        elif fault == "duplicate_frame":
            yield from pairs + pairs[:1]
        else:
            yield from pairs
            (args.run_root / audit.ARMS[0] / "final.pth").write_bytes(b"changed")

    monkeypatch.setattr(audit, "_iter_pairs", changed)
    with pytest.raises(ValueError):
        audit.run(args)
    assert not (args.out_dir / "report.json").exists()


def test_cross_arm_input_identity_mismatch_rejected(fixture):
    _, pairs = fixture
    row = audit.infer_rows(IdentityModel(), pairs[:1], "cpu")[0]
    other = copy.deepcopy(row)
    other["array_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="arm observations differ"):
        audit.paired_key(row, other)
