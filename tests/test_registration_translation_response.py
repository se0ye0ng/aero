import pytest
import torch

from scripts.probe_registration_translation_response import response, shifted_image


@pytest.mark.parametrize("dx,dy", [(4, 0), (-4, 0), (0, 4), (0, -4)])
def test_known_translation_sign_units_and_no_response_reference(dx, dy):
    image = torch.zeros(1, 3, 32, 64)
    image[..., 16, 32] = 1
    shifted = shifted_image(image, dx, dy)
    assert torch.all(shifted[..., 16 + dy, 32 + dx] == 1)
    assert shifted.sum() == image.sum()
    base = torch.zeros(1, 2, 32, 64)
    delta = torch.tensor([2 * dx / 64, 2 * dy / 32])[None, :, None, None]
    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    ideal = response(base, base + delta, box, dx, dy)
    constant = response(base, base, box, dx, dy)
    for domain in ("global", "target_roi"):
        assert ideal[domain]["median_error_pixels"] == 0
        assert ideal[domain]["mean_projected_response_gain"] == 1
        assert constant[domain]["median_error_pixels"] == 4
        assert constant[domain]["mean_projected_response_gain"] == 0
        assert ideal[domain]["observed_pixels"] == constant[domain]["observed_pixels"]


def test_bad_prediction_cannot_drop_its_errors_by_reducing_support():
    base = torch.zeros(1, 2, 32, 32)
    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    expected = response(base, base, box, 4, 0)
    far = response(base, base + 10, box, 4, 0)
    invalid = response(base, base * float("nan"), box, 4, 0)
    assert far["target_roi"]["observed_pixels"] == expected["target_roi"]["observed_pixels"]
    assert far["target_roi"]["median_error_pixels"] > 4
    assert not invalid["finite_prediction"]
    assert invalid["target_roi"]["median_error_pixels"] is None
