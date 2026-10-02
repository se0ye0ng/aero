import json

import torch

from aero_ir.registration.protocol_v7 import observation
from scripts.probe_registration_v7_mind_gradient import gradient_terms, summarize


def fixture():
    visible = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(8))
    infrared = visible.roll(2, dims=-1)
    vb = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    ib = vb + torch.tensor([[2 / 32, 0, 0, 0]])
    excluded = torch.zeros(1, 32, 32, dtype=torch.bool)
    observations = [observation(visible, vb, excluded), observation(infrared, ib, excluded)]
    return visible, infrared, vb, ib, observations


def test_shared_velocity_gradient_probe_is_finite_without_mutating_input():
    velocity = torch.zeros(1, 2, 32, 32)
    result = gradient_terms(velocity, *fixture())
    assert result["geometry_gradient_norm"] > 0
    assert result["weighted_mind_gradient_norm"] > 0
    assert result["weighted_mind_to_geometry_norm_ratio"] > 0
    assert -1.000001 <= result["gradient_cosine"] <= 1.000001
    assert not velocity.requires_grad and velocity.grad is None
    assert torch.equal(velocity, torch.zeros_like(velocity))
    json.dumps(result, allow_nan=False)


def test_missing_descriptor_evidence_has_no_fake_gradient_direction():
    args = fixture()
    for obs in args[-1]:
        obs["observe"].zero_()
    result = gradient_terms(torch.zeros(1, 2, 32, 32), *args)
    assert result["weighted_mind_gradient_norm"] == 0
    assert result["weighted_mind_to_geometry_norm_ratio"] == 0
    assert result["gradient_cosine"] is None
    summary = summarize([result])
    assert summary["zero_mind_gradient_frames"] == 1
    assert summary["gradient_cosine"]["measured"] == 0
    assert summary["gradient_cosine"]["median"] is None
    json.dumps(summary, allow_nan=False)
