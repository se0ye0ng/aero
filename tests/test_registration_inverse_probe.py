import pytest
import torch

from aero_ir.registration.geometry import centre_grid, cycle_field, pixel_norm
from scripts.probe_registration_inverse import affine_projection, invert_field


def field():
    zero = torch.zeros(1, 2, 64, 64)
    grid = centre_grid(zero)
    mapped = grid * torch.tensor([1.15, 0.95]) + torch.tensor([0.03, -0.02])
    return (mapped - grid).permute(0, 3, 1, 2)


def test_inverse_restores_cycle_on_valid_support():
    forward = field()
    inverse, stats = invert_field(forward, torch.zeros_like(forward))
    residual, valid = cycle_field(forward, inverse)
    assert torch.quantile(pixel_norm(residual, 64, 64)[valid], 0.95) < 1e-3
    assert stats["supported_converged_fraction_0_1px"] > 0.8


def test_affine_projection_recovers_original_without_boxes():
    original = field()
    forward, reverse = affine_projection(original)
    assert torch.allclose(forward, original, atol=1e-6)
    residual, valid = cycle_field(forward, reverse)
    assert pixel_norm(residual, 64, 64)[valid].max() < 1e-3


def test_orientation_reversing_affine_rejected():
    original = torch.zeros(1, 2, 64, 64)
    original[:, 0] = -2 * centre_grid(original)[..., 0]
    with pytest.raises(ValueError, match="orientation"):
        affine_projection(original)


def test_singular_field_has_finite_inverse_but_not_fake_convergence():
    zero = torch.zeros(1, 2, 64, 64)
    collapsed = -centre_grid(zero).permute(0, 3, 1, 2)
    inverse, stats = invert_field(collapsed, zero)
    assert torch.isfinite(inverse).all()
    assert stats["supported_converged_fraction_0_1px"] < 0.01
