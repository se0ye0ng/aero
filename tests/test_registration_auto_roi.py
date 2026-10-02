import numpy as np
import pytest

from scripts.probe_registration_auto_roi import auto_hud, crop_region, prepare, restore_native


def test_coordinate_roundtrip():
    image = np.full((100, 200), 100, np.uint8)
    box = np.array([60.0, 30.0, 100.0, 55.0])
    _, cropped, meta = crop_region(image, box)
    assert np.allclose(restore_native(cropped.reshape(2, 2), meta), box.reshape(2, 2))


def test_clipped_border_crop_roundtrip():
    image = np.full((100, 200), 100, np.uint8)
    box = np.array([0.0, 70.0, 20.0, 99.0])
    _, cropped, meta = crop_region(image, box)
    assert meta["bounds_xyxy"][0] == 0
    assert np.allclose(restore_native(cropped.reshape(2, 2), meta), box.reshape(2, 2))


def test_invalid_box():
    with pytest.raises(ValueError):
        crop_region(np.zeros((50, 50), np.uint8), [20, 20, 10, 30])


def test_hud_header_and_line_not_compact_target():
    gray = np.full((200, 320), 100, np.uint8)
    gray[100, 100:180] = 255
    gray[140:150, 155:165] = 255
    mask = auto_hud(gray)
    assert mask[:40].all()
    assert mask[100, 110:170].all()
    assert not mask[140:150, 155:165].any()


def test_constant_image_does_not_mask_entire_field():
    for v in (0, 100, 255):
        mask = auto_hud(np.full((200, 320), v, np.uint8))
        assert not mask[40:].any()


def test_prepare_shared_support_and_no_inpainting_without_mask():
    g = np.full((200, 320), 100, np.uint8)
    data = prepare(g, [150, 100, 170, 120], 256)
    assert np.array_equal(data["raw"], data["inpaint"])
    assert not data["excluded"].any()
    assert data["meta"]["target_excluded_fraction"] == 0


def test_empty_points_restore():
    _, _, meta = crop_region(np.zeros((50, 50), np.uint8), [10, 10, 20, 20])
    assert restore_native(np.empty((0, 2)), meta).shape == (0, 2)
