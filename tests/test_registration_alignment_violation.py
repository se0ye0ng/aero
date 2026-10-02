from pathlib import Path

import pytest
import torch
import yaml

from aero_ir.registration.alignment_violation import alignment_violation, box_violations
from aero_ir.registration.qualification_v4 import THRESHOLDS, _box_statistics


@pytest.mark.parametrize("scale", [0.5, 0.8, 1.0, 1.2, 1.6])
@pytest.mark.parametrize("shift", [0.0, 0.02, 0.1])
def test_component_signs_match_frozen_box_predicates(scale, shift):
    truth = torch.tensor([[0.5, 0.5, 0.2, 0.3]], dtype=torch.float64)
    pred = truth.clone()
    pred[:, 2:] *= scale
    pred[:, 0] += shift
    pred.requires_grad_()
    values = box_violations(pred, truth)
    metrics = _box_statistics(pred[0].detach(), truth[0])
    expected = {
        "bbox_iou": metrics["bbox_iou"] < THRESHOLDS["minimum_bbox_iou"],
        "centroid_shift_fraction": (
            metrics["centroid_shift_fraction"] > THRESHOLDS["maximum_centroid_shift_fraction"]
        ),
        "absolute_area_ratio_change": (
            metrics["absolute_area_ratio_change"] > THRESHOLDS["maximum_absolute_area_ratio_change"]
        ),
    }
    assert {k: bool(v > 1e-10) for k, v in values.items()} == expected
    sum(v.sum() for v in values.values()).backward()
    assert torch.isfinite(pred.grad).all()


def test_identity_zero_and_reverse_direction_cannot_be_averaged_away():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    zero = torch.zeros(1, 2, 32, 32, requires_grad=True)
    assert float(alignment_violation((zero, zero), boxes, boxes).detach()) == 0
    shifted = torch.zeros_like(zero)
    shifted[:, 0] = 0.3
    shifted.requires_grad_()
    loss = alignment_violation((zero, shifted), boxes, boxes)
    assert loss > 0
    assert torch.allclose(loss, alignment_violation((shifted, zero), boxes, boxes))
    loss.backward()
    assert torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum() > 0


def test_every_pair_is_retained_in_mean_and_truth_is_not_optimized():
    truth = torch.tensor([[0.5, 0.5, 0.2, 0.2]], requires_grad=True)
    zero = torch.zeros(1, 2, 32, 32)
    shift = zero.clone()
    shift[:, 0] = 0.3
    one = alignment_violation((zero, shift), truth, truth)
    both = alignment_violation(
        (torch.cat([zero, zero]), torch.cat([shift, zero])),
        truth.detach().repeat(2, 1),
        truth.detach().repeat(2, 1),
    )
    assert torch.allclose(both, one / 2)
    pred = torch.tensor([[0.7, 0.5, 0.2, 0.2]], requires_grad=True)
    sum(v.sum() for v in box_violations(pred, truth).values()).backward()
    assert pred.grad is not None and truth.grad is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.2])
def test_invalid_truth_rejected(bad):
    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    truth = pred.clone()
    truth[0, 2] = bad
    with pytest.raises(ValueError):
        box_violations(pred, truth)


def test_config_and_schedule_cover_all_fit_pairs_equally():
    from scripts.probe_registration_alignment_violation import read_config, schedule

    cfg = read_config(Path("configs/experiment/registration_alignment_violation_cpu.yaml"))
    ids = [f"train_{i:03}" for i in range(160)]
    order = schedule(ids, cfg["updates_per_arm"], cfg["batch_size"], cfg["seed"])
    assert len(order) == 200
    assert order == schedule(list(reversed(ids)), 200, 8, 0)
    for start in range(0, 200, 20):
        assert sorted(s for batch in order[start : start + 20] for s in batch) == ids


@pytest.mark.parametrize(
    "override",
    [{"device": "cuda"}, {"updates_per_arm": 201}, {"seed": True}, {"learning_rate": float("nan")}],
)
def test_config_rejects_unsupported_or_unbalanced_experiment(tmp_path, override):
    from scripts.probe_registration_alignment_violation import read_config

    cfg = yaml.safe_load(
        Path("configs/experiment/registration_alignment_violation_cpu.yaml").read_text()
    )
    cfg.update(override)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        read_config(path)
