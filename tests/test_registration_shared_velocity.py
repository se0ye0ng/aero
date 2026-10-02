import pytest
import torch

from aero_ir.registration.geometry import cycle_field, jacobian_determinant, pixel_norm
from aero_ir.registration.shared_velocity import control_velocity, shared_fields


def test_zero_velocity_is_exact_centre_identity():
    v = torch.zeros(1, 2, 32, 48)
    forward, inverse = shared_fields(v)
    assert torch.equal(forward, v) and torch.equal(inverse, v)
    assert torch.allclose(jacobian_determinant(forward), torch.ones(1, 31, 47), atol=1e-5)


def test_constant_translation_and_inverse_on_valid_support():
    v = torch.zeros(1, 2, 32, 48)
    v[:, 0] = 2 * 3 / 48
    v[:, 1] = -2 * 2 / 32
    forward, inverse = shared_fields(v)
    assert torch.allclose(forward, v) and torch.allclose(inverse, -v)
    residual, valid = cycle_field(forward, inverse)
    assert not valid.all()  # Out-of-camera pixels are not accepted by border extension.
    assert pixel_norm(residual, 32, 48)[valid].max() < 1e-4


def test_smooth_nonlinear_field_and_backward():
    y, x = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij")
    controls = torch.stack((0.2 * y, -0.2 * x))[None].requires_grad_(True)
    velocity = control_velocity(controls, (64, 64), max_pixels=8)
    forward, inverse = shared_fields(velocity)
    assert not torch.allclose(inverse, -forward, atol=1e-4)
    for a, b in ((forward, inverse), (inverse, forward)):
        residual, valid = cycle_field(a, b)
        assert valid.all()
        assert jacobian_determinant(a).min() > 0
        assert pixel_norm(residual, 64, 64).max() < 0.1
    (forward.square().sum() + inverse.square().sum()).backward()
    assert controls.grad is not None and torch.isfinite(controls.grad).all()
    assert controls.grad.abs().sum() > 0


def test_boundary_taper_and_pixel_units():
    controls = torch.full((1, 2, 4, 4), 100.0)
    velocity = control_velocity(controls, (32, 64), max_pixels=4)
    assert torch.count_nonzero(velocity[..., 0, :]) == 0
    assert torch.count_nonzero(velocity[..., -1, :]) == 0
    assert torch.count_nonzero(velocity[..., :, 0]) == 0
    assert torch.count_nonzero(velocity[..., :, -1]) == 0
    assert (velocity[:, 0].abs() <= 8 / 64).all()
    assert (velocity[:, 1].abs() <= 8 / 32).all()


@pytest.mark.parametrize("steps", [0, -1, 1.5, True, 13])
def test_invalid_integrator_rejected(steps):
    with pytest.raises(ValueError):
        shared_fields(torch.zeros(1, 2, 16, 16), steps)


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        shared_fields(torch.full((1, 2, 16, 16), float("nan")))
