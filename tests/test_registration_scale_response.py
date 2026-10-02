import pytest
import torch

from aero_ir.registration.geometry import centre_grid, sampling_map
from scripts.probe_registration_scale_response import (
    SCALES,
    affine,
    scale_response,
    scaled_image,
)


@pytest.mark.parametrize("sx,sy", SCALES)
def test_known_scale_of_translated_field_and_unchanged_reference(sx, sy):
    base = torch.zeros(1, 2, 32, 64, dtype=torch.float64)
    base[:, 0] = 0.05
    base[:, 1] = -0.03
    centre = torch.tensor([0.1, -0.1], dtype=base.dtype)
    box = torch.tensor([[0.5, 0.5, 0.2, 0.3]], dtype=base.dtype)
    expected = (affine(sampling_map(base), centre, sx, sy) - centre_grid(base)).permute(0, 3, 1, 2)
    ideal = scale_response(base, expected, box, centre, sx, sy)
    unchanged = scale_response(base, base, box, centre, sx, sy)
    assert ideal["target_roi"]["p95_error_pixels"] < 1e-12
    assert ideal["mapped_target_box"]["size_ratio_xy"] == pytest.approx([sx, sy])
    assert unchanged["mapped_target_box"]["size_ratio_xy"] == pytest.approx([1, 1])
    for i, scale in enumerate((sx, sy)):
        if scale != 1:
            assert ideal["mapped_target_box"]["log_scale_gain_xy"][i] == pytest.approx(1)
            assert unchanged["mapped_target_box"]["log_scale_gain_xy"][i] == pytest.approx(0)
        else:
            assert ideal["mapped_target_box"]["log_scale_gain_xy"][i] is None


def test_raster_inverse_transport_and_identity_control():
    field = torch.zeros(1, 2, 32, 64, dtype=torch.float64)
    grid = centre_grid(field)
    image = torch.cat((grid.permute(0, 3, 1, 2), torch.ones_like(field[:, :1])), dim=1)
    centre = torch.tensor([0.1, -0.1], dtype=image.dtype)
    assert torch.allclose(scaled_image(image, centre, 1, 1), image, atol=1e-12)
    scaled = scaled_image(image, centre, 1.25, 0.8)
    expected = affine(grid, centre, 1.25, 0.8, inverse=True).permute(0, 3, 1, 2)
    assert torch.allclose(scaled[:, :2, 8:24, 16:48], expected[:, :, 8:24, 16:48], atol=1e-12)


def test_bad_prediction_cannot_choose_support():
    base = torch.zeros(1, 2, 32, 64)
    centre = torch.zeros(2)
    box = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    unchanged = scale_response(base, base, box, centre, 1.25, 0.8)
    invalid = scale_response(base, base * float("nan"), box, centre, 1.25, 0.8)
    far = scale_response(base, base + 10, box, centre, 1.25, 0.8)
    assert far["target_roi"]["observed_pixels"] == unchanged["target_roi"]["observed_pixels"]
    assert invalid["target_roi"]["observed_pixels"] == unchanged["target_roi"]["observed_pixels"]
    assert far["target_roi"]["median_error_pixels"] > 100
    assert invalid["target_roi"]["median_error_pixels"] is None
    assert invalid["mapped_target_box"]["size_ratio_xy"] == [None, None]


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match="positive scales"):
        affine(torch.zeros(1, 2), torch.zeros(2), scale, 1)
