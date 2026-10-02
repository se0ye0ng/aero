"""Verify the v5 repair targets the physical errors that v4's mean can conceal."""

import pytest
import torch

from aero_ir.registration.geometry import centre_grid, from_superfusion
from aero_ir.registration.protocol_v5 import _direction_constraints, bidirectional_geometry_loss
from aero_ir.registration.training_v5 import training_step


def _box():
    return torch.tensor([[0.5, 0.5, 0.1, 0.1]])


def test_identity_has_zero_geometric_repair_loss():
    field = torch.zeros(1, 2, 64, 64)
    terms = _direction_constraints(field, field, _box())
    assert all(value.item() == 0 for value in terms.values())


def test_local_cycle_error_is_not_diluted_by_image_average():
    first = torch.zeros(1, 2, 64, 64)
    first[:, 0, 28:36, 28:36] = 0.25  # eight pixels on a small ROI
    second = torch.zeros_like(first)
    terms = _direction_constraints(first, second, _box())
    assert terms["cycle_global_tail"] > 1
    assert terms["cycle_roi_tail"] > 7
    assert terms["cycle_roi_max"] > 3


def test_perfect_cycle_does_not_hide_reflection():
    first = torch.zeros(1, 2, 64, 64)
    first[:, 0] = -2 * centre_grid(first)[..., 0]
    terms = _direction_constraints(first, first, _box())
    assert terms["cycle_global_tail"] < 1e-4
    assert terms["jacobian_global_tail"] >= 5.9
    assert terms["jacobian_roi"] >= 5.9


def test_no_overlap_has_differentiable_support_penalty():
    first = torch.full((1, 2, 32, 32), 3.0, requires_grad=True)
    second = torch.zeros_like(first)
    terms = _direction_constraints(first, second, _box())
    loss = terms["support_global"] + terms["support_roi"]
    assert loss > 0
    loss.backward()
    assert torch.isfinite(first.grad).all() and first.grad.abs().max() > 0


def test_actual_bidirectional_objective_has_finite_gradients():
    torch.manual_seed(17)
    image = torch.rand(1, 3, 32, 32)
    zero = torch.zeros(1, 2, 32, 32)
    # Raw identity cancels the legacy endpoint lattice exactly once.
    first = (-from_superfusion(zero) + torch.randn_like(zero) * 0.02).requires_grad_()
    second = (-from_superfusion(zero) + torch.randn_like(zero) * 0.02).requires_grad_()
    loss, values = bidirectional_geometry_loss(image, image, _box(), _box(), first, second)
    assert torch.isfinite(loss) and values["cycle_global_tail"] > 0
    loss.backward()
    for field in (first, second):
        assert torch.isfinite(field.grad).all() and field.grad.abs().max() > 0


class _FieldMatcher(torch.nn.Module):
    """Expose both raw fields to test the actual optimizer-step contract on CPU."""

    def __init__(self):
        super().__init__()
        identity = -from_superfusion(torch.zeros(1, 2, 32, 32))
        self.forward_field = torch.nn.Parameter(identity + 0.03)
        self.reverse_field = torch.nn.Parameter(identity - 0.01)

    def forward(self, infrared, visible, *, direction):
        if direction == "visible_to_infrared":
            return self.forward_field
        return self.reverse_field


def test_optimizer_step_updates_both_directions_with_finite_diagnostics():
    torch.manual_seed(21)
    model = _FieldMatcher()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image = torch.rand(1, 3, 32, 32)
    values, checks = training_step(model, optimizer, image, image, _box(), _box(), diagnostics=True)
    assert values["total"] > 0
    assert checks["pre_clip_gradient_norm"] > 0
    assert checks["forward_gradient_max_abs"] > 0
    assert checks["reverse_gradient_max_abs"] > 0
    for old, new in zip(before, model.parameters(), strict=True):
        assert torch.isfinite(new).all() and not torch.equal(old, new)


def test_optimizer_rejects_nonfinite_field_before_parameter_update():
    model = _FieldMatcher()
    with torch.no_grad():
        model.forward_field[0, 0, 0, 0] = float("nan")
    reverse_before = model.reverse_field.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image = torch.rand(1, 3, 32, 32)
    with pytest.raises(FloatingPointError, match="invalid forward field"):
        training_step(model, optimizer, image, image, _box(), _box())
    assert torch.equal(reverse_before, model.reverse_field)
