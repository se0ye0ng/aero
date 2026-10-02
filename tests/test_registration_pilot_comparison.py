"""Comparison must use paired frames, real geometry and fail-closed provenance."""

import json
import math
import sys

import numpy as np
import pytest
import torch

from aero_ir.registration.geometry import from_superfusion
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts import compare_antiuav300_registration_pilots as comparison
from scripts.audit_antiuav300_dense_registration import FramePair


class FixedField(torch.nn.Module):
    def __init__(self, shift=0.0):
        super().__init__()
        # Return raw fields whose converted maps are identity or translation.
        field = -from_superfusion(torch.zeros(1, 2, 32, 32))
        field[:, 0] += shift
        self.field = torch.nn.Parameter(field, requires_grad=False)

    def forward(self, infrared, visible, *, direction):
        return self.field.expand(len(visible), -1, -1, -1)


def pairs():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    box = np.array([0.5, 0.5, 0.25, 0.25], dtype=np.float32)
    return [FramePair(s, "sequence", 7, image, image, box, box) for s in ("train", "val")]


def test_paired_geometry_detects_cycle_error_and_exact_coverage():
    expected = {s: {("sequence", 7)} for s in ("train", "val")}
    rows, digest = comparison.evaluate(
        {"v4": FixedField(), "candidate": FixedField(0.1)}, [pairs()], expected
    )
    assert len(digest) == 64
    base, candidate = rows["v4"]["val"], rows["candidate"]["val"]
    stats = comparison.summary_for(base)
    assert stats["joint_frame_pass_rate"] == 1
    assert comparison.summary_for(candidate)["joint_frame_pass_rate"] == 0
    delta = comparison.paired_deltas(candidate, base)["ir_to_rgb_points"]
    assert delta["bbox_iou"]["mean"] < 0
    assert delta["cycle_p95_pixels"]["mean"] == pytest.approx(3.2, abs=1e-4)
    with pytest.raises(ValueError, match="different frame"):
        comparison.paired_deltas(candidate, [])
    with pytest.raises(ValueError, match="duplicate"):
        comparison.evaluate({"v4": FixedField()}, [pairs(), pairs()], expected)
    with pytest.raises(ValueError, match="incomplete"):
        comparison.evaluate({"v4": FixedField()}, [pairs()[:1]], expected)


def test_missing_measurements_are_counted_not_fabricated():
    stats = comparison.summarize([1.0, None, float("nan"), float("inf")])
    assert stats["count"] == 4 and stats["missing_or_nonfinite"] == 3
    assert stats["mean"] == 1
    assert comparison.summarize([None])["median"] is None


def test_inverse_adapter_preserves_native_field_and_reduces_cycle():
    model = FixedField(0.1)
    adapter = comparison.NumericalInverseMatcher(model)
    images = torch.zeros(1, 3, 32, 32)
    assert torch.equal(adapter(images, images, direction="visible_to_infrared"), model.field)
    expected = {s: {("sequence", 7)} for s in ("train", "val")}
    rows, _ = comparison.evaluate({"inverse": adapter}, [pairs()], expected)
    row = rows["inverse"]["val"][0]
    assert row["ir_to_rgb_points"]["cycle_p95_pixels"] < 1e-4
    assert row["rgb_to_ir_points"]["cycle_p95_pixels"] < 1e-4


def save_checkpoint(path, epoch=10, planned=10, access="none", initial=None):
    meta = {
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": access,
        "epochs": planned,
        "cache_manifest_sha256": "same-cache",
        "initial_checkpoint_sha256": initial,
    }
    torch.save({"epoch": epoch, "aero_registration": meta, "DM": {"field": torch.zeros(1)}}, path)


@pytest.mark.parametrize("epoch,planned,access", [(9, 10, "none"), (10, 10, "val")])
def test_reject_incomplete_or_leaky_checkpoint(tmp_path, epoch, planned, access):
    path = tmp_path / "model.pth"
    save_checkpoint(path, epoch, planned, access)
    with pytest.raises(ValueError, match="completed train-only"):
        comparison.checkpoint_info(path)


def test_accept_completed_pilot_without_relaxing_v4_gate(tmp_path):
    from scripts.audit_antiuav300_registration_v4 import checkpoint_provenance

    path = tmp_path / "model.pth"
    save_checkpoint(path)
    assert comparison.checkpoint_info(path)["epoch"] == 10
    with pytest.raises(ValueError, match="300-epoch"):
        checkpoint_provenance(path)


def test_cli_writes_actual_paired_metrics_and_hold(tmp_path, monkeypatch):
    base, candidate = tmp_path / "base.pth", tmp_path / "candidate.pth"
    save_checkpoint(base, epoch=300, planned=300)
    save_checkpoint(candidate, initial=file_sha256(base))
    out = tmp_path / "comparison"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--root",
            str(tmp_path),
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--checkpoint",
            f"v4={base}",
            "--checkpoint",
            f"v6={candidate}",
        ],
    )
    expected = {s: {("sequence", 7)} for s in ("train", "val")}
    monkeypatch.setattr(comparison, "expected_pairs", lambda *args: (expected, {}))
    monkeypatch.setattr(comparison, "_iter_pairs", lambda *args: iter(pairs()))
    monkeypatch.setattr(
        comparison,
        "load_superfusion_matcher",
        lambda path, device: FixedField(0 if path == base else 0.1),
    )
    comparison.main()
    report = json.loads((out / "report.json").read_text())
    signature = report.pop("report_sha256")
    assert signature == canonical_hash(report)
    assert report["data_usage"]["exact_pair_coverage"] is True
    for model in report["models"].values():
        assert model["gates"]["generator_training_eligible"] == "hold"
        assert model["gates"]["exhaustive_geometry"] == "hold"
    assert report["models"]["v4"]["gates"]["geometric_screen"] == "pass"
    assert report["models"]["v6"]["gates"]["geometric_screen"] == "hold"
    mean = report["models"]["v6"]["metrics"]["val"]["directions"]["ir_to_rgb_points"][
        "cycle_p95_pixels"
    ]["mean"]
    assert math.isclose(mean, 3.2, abs_tol=1e-4)
    assert (out / "summary.csv").is_file() and (out / "summary.md").is_file()
    with pytest.raises(SystemExit):
        comparison.main()  # Never overwrite the previous comparison.
