import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from aero_ir.registration.geometry import from_superfusion, map_boxes
from aero_ir.registration.protocol_v7 import (
    SharedVelocityMatcher,
    descriptor_direction,
    observation,
    step,
)
from aero_ir.registration.superfusion import DenseMatcher, load_superfusion_matcher
from scripts.train_antiuav300_registration_v7 import (
    automatic_exclusion,
    epoch_selection,
    load_trained,
    read_batch,
    save_checkpoint,
)


class TranslationPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.translation = torch.nn.Parameter(torch.zeros(1, 2, 1, 1))
        self.calls = 0

    def forward(self, infrared, visible, *, direction):
        assert direction == "visible_to_infrared"
        self.calls += 1
        n, _, h, w = visible.shape
        field = self.translation.expand(n, 2, h, w)
        return field - from_superfusion(torch.zeros_like(field))


def test_one_predictor_call_and_raw_adapter_roundtrip():
    predictor = TranslationPredictor()
    model = SharedVelocityMatcher(predictor)
    image = torch.zeros(1, 3, 32, 32)
    f, r = model.fields(image, image)
    assert predictor.calls == 1
    assert f.abs().max() == 0 and r.abs().max() == 0
    for direction in ("visible_to_infrared", "infrared_to_visible"):
        raw = model(image, image, direction=direction)
        assert from_superfusion(raw).abs().max() == 0


def test_unsupported_source_cannot_reduce_mind_loss():
    image = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
    obs = observation(image, boxes, torch.zeros(1, 32, 32, dtype=torch.bool))
    identity = torch.zeros(1, 2, 32, 32)
    baseline, _ = descriptor_direction(obs, obs, identity)
    missing, counts = descriptor_direction(obs, obs, identity + 5)
    assert baseline < 1e-10
    assert missing == 1 and counts["source_observed_fraction"] == 0
    assert counts["target_observed_pixels"] > 0


def test_empty_texture_explicitly_reports_no_evidence():
    image = torch.zeros(1, 3, 32, 32)
    boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
    obs = observation(image, boxes, torch.zeros(1, 32, 32, dtype=torch.bool))
    loss, counts = descriptor_direction(obs, obs, torch.zeros(1, 2, 32, 32))
    assert loss == 0
    assert counts["eligible_fraction"] == 0 and counts["target_observed_pixels"] == 0


def test_shared_translation_learns_both_directions_on_cpu():
    torch.set_num_threads(1)
    model = SharedVelocityMatcher(TranslationPredictor())
    image = torch.zeros(1, 3, 32, 32)
    target = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    source = torch.tensor([[0.54, 0.48, 0.25, 0.25]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(30):
        values = step(model, optimizer, image, image, source, target)
        assert values["gradient_norm"] >= 0 and torch.isfinite(torch.tensor(values["total"]))
    f, r = model.fields(image, image)
    assert (map_boxes(target, f)[:, :2] - source[:, :2]).abs().max() < 0.01
    assert (map_boxes(source, r)[:, :2] - target[:, :2]).abs().max() < 0.01


def test_selection_is_matched_between_arms_and_covers_sequences():
    shards = [{"sequence_id": "b", "pairs": 100}, {"sequence_id": "a", "pairs": 100}]
    first = epoch_selection(shards, 0, 0)
    assert first == epoch_selection(list(reversed(shards)), 0, 0)
    assert len(first) == len(set(first)) == 32
    assert sum(sid == "a" for sid, _ in first) == 16
    assert first != epoch_selection(shards, 1, 0)


def test_exclusion_header_and_boundary_guard():
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    mask = automatic_exclusion(image)
    assert mask.dtype == bool and mask[:34].all()
    assert not mask[64, 64]


def test_checkpoint_cannot_be_loaded_as_legacy_displacement(tmp_path):
    model = SharedVelocityMatcher(DenseMatcher()).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, model, optimizer, scheduler, 0, {"test": True})
    restored, state = load_trained(path, torch.device("cpu"))
    assert state["completed_epochs"] == 0 and "DM" not in state
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="no DM"):
        load_superfusion_matcher(path, torch.device("cpu"))


def test_batch_content_hash_tracks_consumed_data(tmp_path):
    folder = tmp_path / "shards/sequence"
    folder.mkdir(parents=True)
    for key in ("visible", "infrared"):
        np.save(folder / f"{key}.npy", np.zeros((2, 16, 16, 3), np.uint8))
    for key in ("source_boxes", "target_boxes"):
        np.save(folder / f"{key}.npy", np.ones((2, 4), np.float32))
    data, before = read_batch(tmp_path, [("sequence", 0)])
    assert data["visible"].shape == (1, 16, 16, 3)
    np.save(folder / "visible.npy", np.ones((2, 16, 16, 3), np.uint8))
    _, after = read_batch(tmp_path, [("sequence", 0)])
    assert before != after
    with pytest.raises(ValueError, match="outside cache"):
        read_batch(tmp_path, [("../../escape", 0)])


def test_wrapper_rejects_newline_before_work():
    result = subprocess.run(
        ["bash", "scripts/run_antiuav300_registration_v7.sh"],
        env=dict(os.environ, AERO_V7_OUTPUT="bad\npath"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "newline" in result.stderr


@pytest.mark.parametrize("mode", ["run", "preflight"])
def test_wrapper_refuses_existing_arm_directory(tmp_path, monkeypatch, mode):
    # This path must exit before CPU tests or a CUDA probe. Do not let the
    # parent CPU-only environment's empty GPU mask exercise a different guard.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    (tmp_path / "geometry").mkdir()
    result = subprocess.run(
        ["bash", "scripts/run_antiuav300_registration_v7.sh", "geometry", mode],
        env=dict(os.environ, AERO_V7_OUTPUT=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "Refusing to overwrite" in result.stderr


def test_wrapper_rejects_explicitly_hidden_gpu_before_work(tmp_path):
    result = subprocess.run(
        ["bash", "scripts/run_antiuav300_registration_v7.sh", "geometry"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", AERO_V7_OUTPUT=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES is empty" in result.stderr
    assert "Running CPU" not in result.stdout
    assert not (tmp_path / "geometry").exists()


def test_wrapper_never_overrides_scheduler_gpu_visibility():
    source = Path("scripts/run_antiuav300_registration_v7.sh").read_text()
    assert "CUDA_VISIBLE_DEVICES=0" not in source
