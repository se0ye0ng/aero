import numpy as np
import pytest
import torch

from scripts.probe_dinov2_matching import match_features, metrics, patch_centres, prepare, restore


def test_patch_centres_and_half_pixel_restore():
    image = np.zeros((360, 640, 3), np.uint8)
    prepared, meta = prepare(image, "visible", "header_crop")
    assert prepared.shape == (280, 630, 3)
    p = patch_centres(prepared.shape[:2])
    assert np.allclose(p[0], [6.5, 6.5])
    restored = restore(p, meta)
    assert np.allclose(restored[0], [7 * 640 / 630 - 0.5, 7 * 288 / 280 - 0.5 + 72])
    assert (restored[:, 1] >= 72).all()
    assert (restored[:, 0] < 640).all()
    assert len(p) == 20 * 45


def test_ir_grayscale_repeated_and_header_excluded():
    image = np.zeros((512, 640, 3), np.uint8)
    image[:104] = 255
    image[104:, :, 0] = 200
    prepared, meta = prepare(image, "infrared", "header_crop")
    assert prepared.shape == (406, 630, 3)
    assert meta["crop_top"] == 104
    assert np.array_equal(prepared[..., 0], prepared[..., 1])
    assert np.array_equal(prepared[..., 0], prepared[..., 2])
    assert prepared.max() < 100


def test_mutual_matches_recover_permutation():
    a = torch.eye(5)
    permutation = torch.tensor([2, 4, 0, 1, 3])
    b = a[permutation]
    i, j, score = match_features(a, b)
    assert np.array_equal(permutation[j].numpy(), i)
    assert np.allclose(score, 1)
    ri, rj, _ = match_features(b, a)
    assert set(zip(i, j, strict=True)) == set(zip(rj, ri, strict=True))


def test_zero_and_empty_features_abstain():
    assert len(match_features(torch.zeros(3, 8), torch.zeros(5, 8))[0]) == 0
    assert len(match_features(torch.empty(0, 8), torch.zeros(5, 8))[0]) == 0
    with pytest.raises(ValueError):
        match_features(torch.full((3, 8), float("nan")), torch.zeros(5, 8))


def test_boxes_only_change_evaluation_not_matching():
    a = torch.eye(4)
    i, j, _ = match_features(a, a)
    p = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    box = np.array([0.0, 0.0, 3.0, 3.0])
    first = metrics(p[i], p[j], p, p, [box, box])
    second = metrics(p[i], p[j], p, p, [box + 20, box])
    assert first["both_box_matches"] == 4
    assert first["rgb_hull_fraction"] == pytest.approx(1 / 9)
    assert not first["low_target_resolution"]
    assert second["both_box_matches"] == 0
    assert second["low_target_resolution"]
    assert np.array_equal(match_features(a, a)[0], i)


def test_original_keeps_color_and_non_square_geometry():
    image = np.zeros((360, 640, 3), np.uint8)
    image[..., 0] = 255
    prepared, meta = prepare(image, "visible", "original")
    assert prepared.shape == (350, 630, 3)
    assert (prepared[..., 0] == 255).all()
    assert (prepared[..., 1:] == 0).all()
    assert meta["crop_top"] == 0
