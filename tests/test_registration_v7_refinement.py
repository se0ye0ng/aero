import copy

import pytest
import torch

from scripts.probe_registration_v7_refinement import measurements, refine, select_observations


def test_refinement_fits_translation_without_mutating_inputs():
    torch.set_num_threads(1)
    velocity = torch.zeros(1, 2, 64, 64)
    image = torch.zeros(1, 3, 64, 64)
    vb = torch.tensor([[0.54, 0.5, 0.16, 0.16]])
    ib = torch.tensor([[0.5, 0.5, 0.16, 0.16]])
    inputs = (velocity, image, image, vb, ib)
    originals = [x.clone() for x in inputs]
    result = refine(*inputs, steps=30, grid=8, lr=0.003)
    for direction in ("ir_to_rgb_points", "rgb_to_ir_points"):
        assert result["after"][direction]["bbox_iou"] > result["before"][direction]["bbox_iou"]
    assert len(result["trace"]) == 30
    assert result["final_loss"]["total"] < result["trace"][0]["before_update_loss"]["total"]
    for value, original in zip(inputs, originals, strict=True):
        assert torch.equal(value, original)
        assert value.grad is None


def test_selection_includes_failure_and_success_without_score_ranking():
    zero = torch.zeros(1, 2, 32, 32)
    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    good = measurements((zero, zero), box, box)
    bad = copy.deepcopy(good)
    bad["ir_to_rgb_points"]["bbox_iou"] = 0.1
    rows = [{"sequence_id": "z", **bad}, {"sequence_id": "b", **good}, {"sequence_id": "a", **bad}]
    assert select_observations(rows, 1) == {"failed": ["a"], "passed": ["b"]}
    with pytest.raises(ValueError, match="duplicated"):
        select_observations(rows + [rows[0]], 1)


@pytest.mark.parametrize("steps,grid,lr", [(0, 8, 0.01), (2, 1, 0.01), (2, 8, float("nan"))])
def test_invalid_budget(steps, grid, lr):
    with pytest.raises(ValueError, match="invalid refinement"):
        refine(None, None, None, None, None, steps=steps, grid=grid, lr=lr)
