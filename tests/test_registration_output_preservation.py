import pytest
import torch

from aero_ir.registration.output_preservation import direction_preservation, preservation_loss


def test_exact_preservation_zero_and_teacher_detached():
    teacher = torch.zeros(1, 2, 32, 64, requires_grad=True)
    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    student = torch.zeros_like(teacher, requires_grad=True)
    loss, _ = direction_preservation(student, teacher, box)
    assert float(loss.detach()) == 0
    shifted = torch.full_like(teacher, 0.05, requires_grad=True)
    loss, _ = direction_preservation(shifted, teacher, box)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum() > 0
    corrected = shifted.detach() - 0.001 * shifted.grad
    assert direction_preservation(corrected, teacher, box)[0] < loss


def test_pixel_units_on_non_square_images_and_fixed_teacher_support():
    teacher = torch.zeros(1, 2, 32, 64)
    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    shifted = teacher.clone()
    shifted[:, 0] = 2 / 64  # one x pixel; SmoothL1(1)=.5, then average xy ->.25
    loss, meta = direction_preservation(shifted, teacher, box)
    assert loss.item() == pytest.approx(0.25 * 1.05)
    _, bad = direction_preservation(teacher + 10, teacher, box)
    assert bad["global_observed_pixels"] == meta["global_observed_pixels"]
    assert bad["roi_points_per_observation"] == 81


def test_invalid_teacher_not_used_as_supported_correspondence():
    teacher = torch.zeros(1, 2, 32, 32)
    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    with pytest.raises(ValueError, match="supported ROI"):
        direction_preservation(teacher, teacher + 10, box)
    with pytest.raises(ValueError, match="nonfinite"):
        direction_preservation(teacher * float("nan"), teacher, box)
    with pytest.raises(ValueError, match="both registration"):
        preservation_loss((teacher,), (teacher, teacher), box, box)
