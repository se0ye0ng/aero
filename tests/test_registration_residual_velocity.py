import inspect

import pytest
import torch
from torch import nn

from aero_ir.registration.geometry import from_superfusion
from aero_ir.registration.residual_velocity import (
    ResidualVelocityHead,
    ResidualVelocityMatcher,
    image_context,
)
from aero_ir.registration.shared_velocity import shared_fields
from scripts.probe_registration_residual_learning import fit_head


class ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, ir, vis, *, direction):
        return self.weight.expand(ir.shape[0], 2, *ir.shape[2:])


def test_zero_residual_preserves_base_and_frozen_state():
    torch.set_num_threads(1)
    base = nn.Module()
    base.predictor = ToyPredictor()
    model = ResidualVelocityMatcher(base).train()
    image = torch.rand(1, 3, 32, 48)
    fields = model.fields(image, image)
    expected = shared_fields(
        from_superfusion(base.predictor(image, image, direction="visible_to_infrared"))
    )
    assert all(torch.equal(a, b) for a, b in zip(fields, expected, strict=True))
    assert not base.training and model.head.training
    (fields[0].square().mean() + fields[1].square().mean()).backward()
    assert model.head.output.weight.grad.abs().sum() > 0
    assert base.predictor.weight.grad is None
    assert not base.predictor.weight.requires_grad


def test_context_is_detached_image_conditioned_and_contains_no_box_argument():
    torch.manual_seed(4)
    image = torch.rand(1, 3, 32, 32, requires_grad=True)
    velocity = torch.zeros(1, 2, 32, 32, requires_grad=True)
    context = image_context(image, image, velocity)
    assert context.shape == (1, 14, 32, 32)
    assert not context.requires_grad
    assert list(inspect.signature(image_context).parameters) == [
        "visible",
        "infrared",
        "base_velocity",
    ]
    head = ResidualVelocityHead()
    with torch.no_grad():
        head.output.weight.normal_(std=0.01)
    changed = image_context(image, 1 - image, velocity)
    assert not torch.equal(head(context), head(changed))


def test_context_rejects_out_of_range_images():
    with pytest.raises(ValueError, match="images must"):
        image_context(
            torch.full((1, 3, 32, 32), 2.0), torch.zeros(1, 3, 32, 32), torch.zeros(1, 2, 32, 32)
        )


def test_probe_frames_cannot_enter_fit_optimizer():
    head = ResidualVelocityHead()
    original = {k: v.clone() for k, v in head.state_dict().items()}
    with pytest.raises(ValueError, match="probe observation"):
        fit_head(head, [{"role": "probe"}], steps=1, lr=1e-4)
    assert all(torch.equal(original[k], v) for k, v in head.state_dict().items())
