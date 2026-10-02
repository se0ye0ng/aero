import json

import pytest
import torch

from scripts.compare_registration_residual_pilot import compare
from scripts.train_registration_residual_pilot import (
    ARMS,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_roundtrip_preserves_state_and_optimizer(tmp_path, arm):
    model = build_model(arm)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3200)
    parameter = next(p for p in model.parameters() if p.requires_grad)
    parameter.square().mean().backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "latest.pth"
    save_checkpoint(path, model, optimizer, scheduler, 1, {"arm": arm}, torch.device("cpu"))
    restored, state = load_checkpoint(path, torch.device("cpu"))
    assert state["completed_epochs"] == 1
    assert state["scheduler"]["last_epoch"] == 1
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    resumed_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad])
    resumed_optimizer.load_state_dict(state["optimizer"])
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert not path.with_suffix(".partial").exists()
    if arm == "residual_head":
        assert not any(p.requires_grad for p in restored.base_model.parameters())
        assert all(p.requires_grad for p in restored.head.parameters())


def test_checkpoint_rejects_unrelated_payload(tmp_path):
    path = tmp_path / "old.pth"
    torch.save({"architecture": "v7"}, path)
    with pytest.raises(ValueError, match="not a residual"):
        load_checkpoint(path, torch.device("cpu"))


def test_comparison_rejects_smoke_before_product_verification(tmp_path):
    folder = tmp_path / ARMS[0]
    folder.mkdir()
    (folder / "run_spec.json").write_text(
        json.dumps({"arm": ARMS[0], "epochs": 10, "batch_size": 8, "seed": 0, "smoke_steps": 2})
    )
    with pytest.raises(ValueError, match="not the completed frozen"):
        compare(tmp_path, tmp_path / "unused_cache")


def test_unknown_arm_rejected():
    with pytest.raises(ValueError, match="unknown"):
        build_model("v7")
