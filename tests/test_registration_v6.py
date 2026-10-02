"""Regression preservation must be per-pair and leave the teacher immutable."""

import copy

import pytest
import torch

from aero_ir.registration.geometry import from_superfusion
from aero_ir.registration.protocol_v6 import (
    PRESERVATION_TOLERANCES,
    bidirectional_geometry_loss,
    box_errors,
    preservation_penalty,
)
from aero_ir.registration.training_v6 import training_step


def test_pair_regression_cannot_be_cancelled_by_another_pairs_improvement():
    student = {key: torch.tensor([0.4, 0.0], requires_grad=True) for key in PRESERVATION_TOLERANCES}
    teacher = {key: torch.tensor([0.2, 0.2], requires_grad=True) for key in PRESERVATION_TOLERANCES}
    penalty, terms = preservation_penalty(student, teacher)
    assert penalty > 0 and all(value > 0 for value in terms.values())
    penalty.backward()
    for key in student:
        assert student[key].grad[0] > 0 and student[key].grad[1] == 0
        assert teacher[key].grad is None


def test_same_teacher_has_zero_preservation_penalty():
    errors = {key: torch.tensor([0.1, 0.3]) for key in PRESERVATION_TOLERANCES}
    loss, _ = preservation_penalty(errors, errors)
    assert loss == 0


def test_box_errors_convert_raw_identity_once():
    raw = -from_superfusion(torch.zeros(2, 2, 32, 32))
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
    values = box_errors(boxes, boxes, raw)
    for value in values.values():
        assert torch.allclose(value, torch.zeros(2), atol=1e-5)


def test_v6_objective_gradients_and_teacher_detachment():
    torch.manual_seed(12)
    image = torch.rand(1, 3, 32, 32)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    identity = -from_superfusion(torch.zeros(1, 2, 32, 32))
    teacher = identity.clone().requires_grad_()
    first = (identity + 0.03).requires_grad_()
    second = (identity - 0.02).requires_grad_()
    loss, values = bidirectional_geometry_loss(
        image, image, boxes, boxes, first, second, teacher, teacher
    )
    assert values["preserve_box_iou"] > 0
    loss.backward()
    assert teacher.grad is None
    for field in (first, second):
        assert torch.isfinite(field.grad).all() and field.grad.abs().max() > 0


class FieldMatcher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        identity = -from_superfusion(torch.zeros(1, 2, 32, 32))
        self.first = torch.nn.Parameter(identity + 0.03)
        self.second = torch.nn.Parameter(identity - 0.01)

    def forward(self, infrared, visible, *, direction):
        return self.first if direction == "visible_to_infrared" else self.second


def test_step_updates_student_and_keeps_teacher_immutable():
    model = FieldMatcher()
    teacher = copy.deepcopy(model).requires_grad_(False)
    before = [parameter.detach().clone() for parameter in teacher.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image = torch.rand(1, 3, 32, 32)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    _, checks = training_step(model, teacher, optimizer, image, image, boxes, boxes)
    assert checks["pre_clip_gradient_norm"] > 0
    for old, reference, updated in zip(
        before, teacher.parameters(), model.parameters(), strict=True
    ):
        assert torch.equal(old, reference) and reference.grad is None
        assert not torch.equal(old, updated) and torch.isfinite(updated).all()


def test_step_rejects_nonfinite_teacher_without_updating_student():
    model = FieldMatcher()
    teacher = copy.deepcopy(model).requires_grad_(False)
    teacher.first[0, 0, 0, 0] = float("nan")
    before = model.first.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image = torch.rand(1, 3, 32, 32)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    with pytest.raises(FloatingPointError, match="invalid v6"):
        training_step(model, teacher, optimizer, image, image, boxes, boxes)
    assert torch.equal(before, model.first)
