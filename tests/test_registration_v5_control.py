"""Exercise the matched-control CLI through real CPU v4 optimizer updates."""

import json
import sys

import numpy as np
import torch

from aero_ir.registration.geometry import from_superfusion
from aero_ir.registration.training_v4 import training_step
from scripts import train_antiuav300_registration_v5_control as control


class _Matcher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        identity = -from_superfusion(torch.zeros(1, 2, 32, 32))
        self.forward_field = torch.nn.Parameter(identity + 0.01)
        self.reverse_field = torch.nn.Parameter(identity - 0.01)

    def forward(self, infrared, visible, *, direction):
        field = self.forward_field if direction == "visible_to_infrared" else self.reverse_field
        return field.expand(len(visible), -1, -1, -1)


def test_matched_control_cpu_run_saves_v4_objective_and_updates(tmp_path, monkeypatch):
    checkpoint = tmp_path / "initial.pth"
    checkpoint.touch()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    output = tmp_path / "output"
    real_hash = control.file_sha256
    monkeypatch.setattr(
        control,
        "file_sha256",
        lambda path: control.INITIAL_CHECKPOINT_SHA256 if path == checkpoint else real_hash(path),
    )
    monkeypatch.setattr(
        control,
        "_build_cache",
        lambda *args: {
            "fit_split": "train",
            "validation_or_test_access": "none",
            "shards": [{}],
            "cache_manifest_sha256": "test-cache",
        },
    )
    rng = np.random.default_rng(7)
    images = rng.integers(0, 255, (16, 32, 32, 3), dtype=np.uint8)
    boxes = np.tile(np.asarray([0.5, 0.5, 0.1, 0.1], dtype=np.float32), (16, 1))
    monkeypatch.setattr(control, "_epoch_arrays", lambda *args: (images, images, boxes, boxes))
    model = _Matcher().eval()
    before = model.forward_field.detach().clone()
    monkeypatch.setattr(control, "load_superfusion_matcher", lambda *args: model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control",
            "--root",
            str(tmp_path),
            "--initial-checkpoint",
            str(checkpoint),
            "--cache-root",
            str(cache),
            "--output-dir",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
        ],
    )
    assert control.training_step is training_step
    control.main()
    saved = torch.load(output / "antiuav300_registration_v5_control_e1.pth", weights_only=True)
    assert saved["epoch"] == 1
    assert saved["aero_registration"]["optimizer_steps"] == 2
    assert saved["aero_registration"]["model_training_mode"] is False
    assert saved["aero_registration"]["generator_training_eligible"] == "hold"
    assert not torch.equal(saved["DM"]["forward_field"], before)
    log = json.loads((output / "train_log.jsonl").read_text())
    assert "inverse_consistency" in log["loss"]
    assert "cycle_global_tail" not in log["loss"]
