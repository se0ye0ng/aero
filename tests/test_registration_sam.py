import numpy as np

from scripts.probe_registration_sam import align, compare_masks, extract_masks, mask_iou


def test_empty_masks_do_not_pass():
    mask = np.zeros((32, 32), bool)
    assert compare_masks(mask, mask, np.eye(3)) is None
    assert mask_iou(mask, mask) is None


def test_known_shift_and_analytic_inverse():
    mask = np.zeros((64, 64), bool)
    mask[20:30, 24:35] = True
    mask[30:40, 24:28] = True
    target = np.roll(mask, 4, axis=1)
    result = align([mask] * 3, [target] * 3)
    assert result["selected"]["dice_loss"] == 0
    assert result["selected_spec"]["dx"] == 4
    assert result["determinant"] > 0
    assert np.allclose(
        np.array(result["matrix_rgb_crop_to_ir_crop"]) @ result["inverse_matrix"], np.eye(3)
    )


def test_prompt_instability_rejects_without_human():
    class Predictor:
        count = 0

        def set_image(self, image):
            pass

        def predict(self, **kwargs):
            mask = np.zeros((1, 64, 64), bool)
            mask[:, 20:30, 20 + 10 * self.count : 30 + 10 * self.count] = True
            self.count += 1
            return mask, [0.99], None

    data = {
        "inpaint": np.zeros((64, 64), np.uint8),
        "box": [18, 18, 32, 32],
        "excluded": np.zeros((64, 64), bool),
    }
    _, stats = extract_masks(Predictor(), data)
    assert not stats["eligible"]
    assert "prompt_jitter_iou_below_0_8" in stats["reasons"]


def test_clipped_foreground_rejects():
    mask = np.zeros((64, 64), bool)
    mask[0:8, 20:30] = True
    assert compare_masks(mask, mask, np.eye(3)) is None
