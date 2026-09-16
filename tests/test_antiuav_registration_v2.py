import numpy as np
import torch

from aero_ir.registration.protocol_v2 import (
    EPOCHS,
    PAIRS_PER_SEQUENCE_PER_EPOCH,
    geometry_first_loss,
    rotating_positions,
)


def test_rotating_selection_covers_every_pair_without_validation_access():
    length = 1000
    selections = [rotating_positions(length, epoch, "train-sequence-0") for epoch in range(63)]
    assert all(len(values) == PAIRS_PER_SEQUENCE_PER_EPOCH for values in selections)
    assert len(np.unique(np.concatenate(selections))) == length
    assert np.array_equal(selections[0], rotating_positions(length, 0, "train-sequence-0"))
    assert EPOCHS == 300


def test_geometry_first_loss_is_finite_and_box_sensitive():
    visible = torch.rand(2, 3, 256, 256)
    infrared = visible.clone()
    boxes = torch.tensor([[0.5, 0.5, 0.08, 0.06], [0.3, 0.7, 0.04, 0.04]])
    zero_flow = torch.zeros(2, 2, 256, 256, requires_grad=True)

    matching_loss, matching = geometry_first_loss(visible, infrared, boxes, boxes, zero_flow)
    matching_loss.backward()

    assert torch.isfinite(matching_loss)
    assert matching["box_iou"] < 1e-5
    assert zero_flow.grad is not None and torch.isfinite(zero_flow.grad).all()

    shifted_boxes = boxes + torch.tensor([0.05, 0.0, 0.0, 0.0])
    shifted_flow = torch.zeros(2, 2, 256, 256)
    _, shifted = geometry_first_loss(visible, infrared, boxes, shifted_boxes, shifted_flow)
    assert shifted["box_iou"] > matching["box_iou"]
    assert shifted["worst_perimeter_fraction"] > matching["worst_perimeter_fraction"]
