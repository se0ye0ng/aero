"""CPU tests for the bounded GPU smoke's shared optimizer path and fail-closed CLI."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

import aero_ir.registration.training_v4 as training
import scripts.smoke_antiuav300_registration_v4 as smoke
from aero_ir.utils.manifest import canonical_hash


class TinyMatcher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_shift = torch.nn.Parameter(torch.tensor([0.01, -0.01]))
        self.reverse_shift = torch.nn.Parameter(torch.tensor([-0.02, 0.03]))

    def forward(self, infrared, visible, *, direction):
        shift = self.forward_shift if direction == "visible_to_infrared" else self.reverse_shift
        return shift[None, :, None, None].expand(visible.shape[0], 2, *visible.shape[2:])


def _inputs():
    torch.manual_seed(2)
    images = (torch.rand(1, 3, 24, 32), torch.rand(1, 3, 24, 32))
    boxes = torch.tensor([[0.5, 0.5, 0.15, 0.12]])
    return (*images, boxes, boxes)


def test_shared_optimizer_step_updates_both_directions_and_finite_adam_state():
    model = TinyMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        values, checks = training.training_step(model, optimizer, *_inputs(), diagnostics=True)
        assert values["total"] > 0
        assert checks["forward_gradient_max_abs"] > 0
        assert checks["reverse_gradient_max_abs"] > 0
        assert checks["parameter_update_max_abs"] > 0
        assert checks["parameters_finite"] and checks["optimizer_state_finite"]
    assert all(not torch.equal(p, old) for p, old in zip(model.parameters(), before, strict=True))


def test_full_and_smoke_optimization_steps_match():
    first, second = TinyMatcher(), TinyMatcher()
    opt1 = torch.optim.AdamW(first.parameters(), lr=5e-5)
    opt2 = torch.optim.AdamW(second.parameters(), lr=5e-5)
    inputs = _inputs()
    values1, empty = training.training_step(first, opt1, *inputs)
    values2, checks = training.training_step(second, opt2, *inputs, diagnostics=True)
    assert values1 == values2 and empty == {} and checks
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )


def test_nonfinite_loss_fails_before_optimizer_update(monkeypatch):
    model = TinyMatcher()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    monkeypatch.setattr(
        training,
        "bidirectional_geometry_loss",
        lambda *args: (torch.tensor(float("nan")), {"total": float("nan")}),
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        training.training_step(model, optimizer, *_inputs(), diagnostics=True)
    assert not optimizer.state
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before, strict=True))


def test_zero_learning_rate_fails_update_check():
    model = TinyMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)
    with pytest.raises(RuntimeError, match="parameter update"):
        training.training_step(model, optimizer, *_inputs(), diagnostics=True)


def test_loader_uses_train_only_and_closes_last_video(monkeypatch, tmp_path):
    calls = []

    def pairs(root, splits, samples):
        calls.append((root, splits, samples))
        try:
            for i in range(4):
                yield SimpleNamespace(split="train", sequence_id=str(i), frame_index=0)
        finally:
            calls.append("closed")

    monkeypatch.setattr(smoke, "_iter_pairs", pairs)
    result = smoke.load_smoke_pairs(tmp_path, 3)
    assert len(result) == 3
    assert calls == [(tmp_path, ("train",), 1), "closed"]
    with pytest.raises(ValueError, match="expected 8"):
        smoke.load_smoke_pairs(tmp_path, 8)


def _argv(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"placeholder; not loaded in CLI failure tests")
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke",
            "--root",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
        ],
    )
    return output


def test_no_cuda_writes_failure_report_without_touching_checkpoint(tmp_path, monkeypatch):
    output = _argv(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original = (tmp_path / "weights.pth").read_bytes()
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code == 1
    report = json.loads((output / "smoke_report.json").read_text())
    digest = report.pop("report_sha256")
    assert digest == canonical_hash(report)
    assert not report["ok"] and report["steps"] == []
    assert "CUDA unavailable" in report["error"]["message"]
    assert report["generator_training_eligible"] == "hold"
    assert (tmp_path / "weights.pth").read_bytes() == original


def test_existing_output_refused_before_gpu_or_data_access(tmp_path, monkeypatch):
    output = _argv(tmp_path, monkeypatch)
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    monkeypatch.setattr(smoke, "run_smoke", lambda *args: pytest.fail("must not execute"))
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code == 2
    assert (output / "keep.txt").read_text() == "keep"


def test_smoke_cli_success_is_not_registration_qualification(tmp_path, monkeypatch):
    output = _argv(tmp_path, monkeypatch)

    def successful_step_fixture(args, report):
        report.update(ok=True, engineering_smoke="pass")

    monkeypatch.setattr(smoke, "run_smoke", successful_step_fixture)
    smoke.main()
    report = json.loads((output / "smoke_report.json").read_text())
    assert report["ok"]
    assert not report["checkpoint_saved"]
    assert report["generator_training_eligible"] == "hold"
