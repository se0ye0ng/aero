import torch
from torch import nn

from aero_ir.registration.superfusion import (
    DenseMatcher,
    transform_boxes_source_to_target,
    warp_source_to_target,
)


def test_dense_matcher_has_official_checkpoint_parameter_layout():
    model = DenseMatcher()
    state = model.state_dict()

    assert len(state) == 62
    assert "feature_extractor_unshare1.layers.0.model.0.weight" in state
    assert "matcher2.layers.3.model.0.weight" in state
    assert "refiner.estimator.3.model.0.weight" in state
    assert not any(key.endswith("grid_down") or key.endswith("scale") for key in state)


def test_zero_displacement_preserves_images_and_subpixel_boxes():
    source = torch.ones(2, 3, 32, 32)
    displacement = torch.zeros(2, 2, 32, 32)
    boxes = torch.tensor(
        [
            [0.41, 0.52, 0.033, 0.019],
            [0.74, 0.28, 0.12, 0.08],
        ]
    )

    warped = warp_source_to_target(source, displacement)
    transformed, residual = transform_boxes_source_to_target(boxes, displacement)

    # The checkpoint was trained with an endpoint grid and align_corners=False.
    # Preserve that upstream convention: the interior is unchanged and the
    # outermost samples receive zero-padding interpolation.
    assert torch.allclose(warped[:, :, 1:-1, 1:-1], source[:, :, 1:-1, 1:-1], atol=1e-6)
    assert torch.allclose(warped[:, :, 0, 0], torch.full((2, 3), 0.25), atol=1e-6)
    assert torch.allclose(transformed, boxes, atol=1e-6)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_box_inverse_solver_recovers_constant_target_to_source_shift():
    displacement = torch.zeros(1, 2, 64, 64)
    displacement[:, 0] = 0.10  # grid coordinates: +0.05 in normalized [0, 1]
    displacement[:, 1] = -0.04  # grid coordinates: -0.02 in normalized [0, 1]
    source_box = torch.tensor([[0.50, 0.50, 0.20, 0.10]])

    transformed, residual = transform_boxes_source_to_target(source_box, displacement)

    expected = torch.tensor([[0.45, 0.52, 0.20, 0.10]])
    assert torch.allclose(transformed, expected, atol=1e-6)
    assert residual.item() < 1e-6


def test_reverse_direction_preserves_modality_specific_encoders(monkeypatch):
    class Multiply(nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor

        def forward(self, value):
            return value * self.factor

    model = DenseMatcher()
    model.feature_extractor_unshare1 = Multiply(2.0)  # infrared encoder
    model.feature_extractor_unshare2 = Multiply(3.0)  # visible encoder
    model.feature_extractor_share1 = nn.Identity()
    model.feature_extractor_share2 = nn.Identity()
    model.feature_extractor_share3 = nn.Identity()

    def fake_match(*features):
        return features[0][:, :2]

    monkeypatch.setattr(model, "_match", fake_match)
    infrared = torch.ones(1, 3, 8, 8)
    visible = torch.full((1, 3, 8, 8), 10.0)

    ir_to_visible = model(infrared, visible, direction="infrared_to_visible")
    visible_to_ir = model(infrared, visible, direction="visible_to_infrared")

    assert torch.equal(ir_to_visible, torch.full((1, 2, 8, 8), 2.0))
    assert torch.equal(visible_to_ir, torch.full((1, 2, 8, 8), 30.0))
