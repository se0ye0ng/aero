import pytest
import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import sampling_map
from aero_ir.registration.mind import masked_descriptor_ssd, mind_descriptor
from scripts.probe_registration_mind import (
    best_index,
    candidates,
    common_support,
    real_check,
    roi_mask,
    score_candidates,
    synthetic_check,
)


def textured_image():
    generator = torch.Generator().manual_seed(73)
    return F.avg_pool2d(torch.rand(1, 3, 64, 64, generator=generator), 3, 1, 1)


def test_descriptor_shape_range_and_affine_contrast_invariance():
    image = textured_image()
    original, variance = mind_descriptor(image)
    changed, _ = mind_descriptor(0.8 - 0.6 * image)
    assert original.shape == (1, 8, 64, 64)
    assert variance.shape == (1, 1, 64, 64)
    assert original.min() >= 0 and original.max() <= 1
    assert torch.allclose(original[:, :, 3:-3, 3:-3], changed[:, :, 3:-3, 3:-3], atol=2e-5)


def test_flat_descriptor_is_finite_but_scoring_abstains():
    image = torch.ones(1, 3, 64, 64) * 0.5
    descriptor, variance = mind_descriptor(image)
    assert torch.isfinite(descriptor).all()
    assert variance.max() == 0
    maps, _ = candidates(torch.zeros(1, 2, 64, 64), torch.zeros(1, 1, 1, 2))
    scores, counts, _ = score_candidates(
        image, image, maps, {"all": torch.ones(64, 64, dtype=torch.bool)}
    )
    assert counts["all"] == 0
    assert scores["all"]["mind"] == []
    assert best_index([], 0) is None
    assert best_index([0.5, 0.5], 0) is None


def test_recovers_known_shift_despite_contrast_inversion():
    result = synthetic_check(textured_image())
    assert result["supported_pixels"] >= 32
    assert result["mind"]["exact_recovery"]
    assert result["mind"]["loss_at_truth"] < 1e-8
    assert result["edge_ncc"]["exact_recovery"]


def test_common_mask_is_supported_for_every_candidate():
    field = torch.zeros(1, 2, 64, 64)
    maps, _ = candidates(field, torch.zeros(1, 1, 1, 2))
    mask = common_support(maps)
    assert mask.any() and not mask.all()
    assert (maps[:, mask].abs() <= 1 - 7 / 64).all()
    assert torch.equal(mask, common_support(maps.flip(0)))


def test_descriptor_loss_has_finite_nonzero_field_gradient():
    image = textured_image()
    descriptor, _ = mind_descriptor(image)
    field = torch.full((1, 2, 64, 64), 0.02, requires_grad=True)
    mask = torch.zeros(1, 64, 64, dtype=torch.bool)
    mask[:, 12:-12, 12:-12] = True
    loss = masked_descriptor_ssd(descriptor, descriptor, field, mask).mean()
    loss.backward()
    assert torch.isfinite(field.grad).all()
    assert field.grad.abs().sum() > 0


def test_ssd_rejects_empty_support():
    descriptor, _ = mind_descriptor(textured_image())
    with pytest.raises(ValueError, match="empty"):
        masked_descriptor_ssd(
            descriptor,
            descriptor,
            torch.zeros(1, 2, 64, 64),
            torch.zeros(1, 64, 64, dtype=torch.bool),
        )


def test_translation_sign_and_pixel_centre_convention():
    field = torch.zeros(1, 2, 64, 64)
    maps, specs = candidates(field, torch.zeros(1, 1, 1, 2), scales=(1.0,))
    i = specs.index({"scale": 1.0, "dx_px": 4, "dy_px": -4})
    delta = maps[i : i + 1] - sampling_map(field)
    assert torch.allclose(delta, torch.tensor([0.125, -0.125]))


def test_source_box_gt_does_not_select_visual_candidate():
    image = textured_image()
    box = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    wrong_box = torch.tensor([[0.65, 0.55, 0.3, 0.3]])
    field = torch.zeros(1, 2, 64, 64)
    kwargs = {"scales": (1.0,), "offsets": (-1, 0, 1)}
    a = real_check(image, image, box, box, field, **kwargs)
    b = real_check(image, image, wrong_box, box, field, **kwargs)
    assert a["losses"] == b["losses"]
    assert a["baseline_iou"] > b["baseline_iou"]
    for region in a["selection"]:
        for method in ("mind", "edge_ncc"):
            assert (
                a["selection"][region][method]["selected"]
                == (b["selection"][region][method]["selected"])
            )
    assert a["selection"]["roi"]["mind"]["selected"] == {
        "scale": 1.0,
        "dx_px": 0,
        "dy_px": 0,
    }


def test_exact_box_region_excludes_expanded_context():
    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    core = roi_mask(box, 64, 64, mode="box")
    expanded = roi_mask(box, 64, 64)
    assert core.sum() == 16 * 16
    assert expanded.sum() == 32 * 32
    assert (core & ~expanded).sum() == 0
    with pytest.raises(ValueError, match="unknown ROI"):
        roi_mask(box, 64, 64, mode="not_a_region")


@pytest.mark.parametrize("image", [torch.zeros(1, 2, 32, 32), torch.zeros(1, 1, 4, 4)])
def test_rejects_invalid_shape(image):
    with pytest.raises(ValueError):
        mind_descriptor(image)
