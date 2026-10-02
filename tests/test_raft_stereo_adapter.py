import numpy as np
import pytest
import torch

from aero_ir.registration.raft_stereo import consistency, infer_disparity, rectify_images
from aero_ir.registration.temporal_stereo import rectification


def test_positive_disparity_and_reverse_consistency_keep_only_supported_coordinates():
    d = np.full((8, 20), 4.0)
    support = np.ones(d.shape, bool)
    result = consistency(d, d, support, support)
    assert not result["valid"][:, :4].any()
    assert result["valid"][:, 4:].all()


def test_nonfinite_nonpositive_out_of_range_and_inconsistent_disparities_are_invalid():
    d = np.full((8, 160), 4.0)
    support = np.ones(d.shape, bool)
    d[0, 20:24] = [np.nan, 0.0, -1.0, 128.0]
    reverse = np.full(d.shape, 4.0)
    reverse[1, 16:18] = 7.0
    result = consistency(d, reverse, support, support)
    assert not result["valid"][0, 20:24].any()
    assert not result["valid"][1, 20]


def test_remapping_masks_rgb_and_gray_are_identical():
    k = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 48.0], [0.0, 0.0, 1.0]])
    b = np.eye(4)
    b[0, 3] = -0.3
    g = rectification(k, k, b, (96, 320))
    gray = np.arange(96 * 320, dtype=np.uint8).reshape(96, 320)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    a, am = rectify_images(gray, gray, g)
    b, bm = rectify_images(rgb, rgb, g)
    np.testing.assert_array_equal(a[0], b[0][..., 0])
    np.testing.assert_array_equal(am[0], bm[0])
    assert not am[0][0].any()


def test_adapter_restores_padding_shape_and_changes_flow_sign_without_rescaling():
    class Padder:
        def __init__(self, shape, divis_by):
            assert divis_by == 32
            self.shape = shape

        def pad(self, *images):
            return [torch.nn.functional.pad(x, (0, 32, 0, 32)) for x in images]

        def unpad(self, flow):
            return flow[..., :-32, :-32]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, a, b, iters, test_mode):
            assert a.shape == (1, 3, 96, 128) and iters == 32 and test_mode
            return None, torch.full((1, 1, 96, 128), -12.0)

    image = np.full((64, 96), 128, np.uint8)
    result = infer_disparity(Model(), Padder, image, image)
    np.testing.assert_array_equal(result, np.full((64, 96), 12.0))


def test_invalid_input_type_is_rejected_before_model_call():
    with pytest.raises(ValueError, match="uint8"):
        infer_disparity(None, None, np.zeros((64, 96)), np.zeros((64, 96)))
