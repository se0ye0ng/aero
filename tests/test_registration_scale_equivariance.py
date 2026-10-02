import pytest
import torch

from aero_ir.registration.geometry import centre_grid, sampling_map
from aero_ir.registration.scale_equivariance import source_scale_loss


def inputs():
    base = torch.zeros(2, 2, 32, 64, dtype=torch.float64, requires_grad=True)
    boxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]] * 2, dtype=base.dtype)
    centres = torch.zeros(2, 2, dtype=base.dtype)
    scales = torch.tensor([[1.25, 0.8], [0.8, 1.25]], dtype=base.dtype)
    return base, boxes, centres, scales


def test_ideal_zero_and_teacher_detached_with_useful_gradient():
    base, boxes, centres, scales = inputs()
    expected = (
        (sampling_map(base).detach() * scales[:, None, None] - centre_grid(base))
        .permute(0, 3, 1, 2)
        .requires_grad_()
    )
    loss, info = source_scale_loss(base, expected, boxes, centres, scales)
    assert loss.item() == pytest.approx(0, abs=1e-20)
    assert info["roi_support_fraction"] == [1, 1]
    unchanged = torch.zeros_like(base, requires_grad=True)
    loss, _ = source_scale_loss(base, unchanged, boxes, centres, scales)
    loss.backward()
    assert base.grad is None
    assert torch.isfinite(unchanged.grad).all() and unchanged.grad.abs().sum() > 0
    moved = unchanged.detach() - 0.001 * unchanged.grad
    next_loss, _ = source_scale_loss(base, moved, boxes, centres, scales)
    assert next_loss < loss


def test_prediction_cannot_change_loss_support_and_missing_support_fails():
    base, boxes, centres, scales = inputs()
    _, first = source_scale_loss(base, torch.zeros_like(base), boxes, centres, scales)
    loss, far = source_scale_loss(base, torch.ones_like(base) * 10, boxes, centres, scales)
    assert first == far and loss > 100
    with pytest.raises(ValueError, match="no measurable"):
        source_scale_loss(base + 10, base, boxes, centres, scales)


def test_nonfinite_or_negative_scale_rejected():
    base, boxes, centres, scales = inputs()
    with pytest.raises(ValueError, match="positive"):
        source_scale_loss(base, base, boxes, centres, -scales)
    with pytest.raises(ValueError, match="nonfinite"):
        source_scale_loss(base, base * float("nan"), boxes, centres, scales)
