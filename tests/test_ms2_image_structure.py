import numpy as np
import torch
from torch.nn import functional as F

from scripts.probe_ms2_image_structure import eroded_common_support, image_scores, sampling_grid


def test_native_sampling_grid_identity_and_halfpixel_units():
    yy, xx = np.indices((12, 16))
    source = np.c_[xx.ravel(), yy.ravel()]
    grid = sampling_grid(source, (12, 16), (12, 16))
    image = torch.arange(192, dtype=torch.float32).reshape(1, 1, 12, 16)
    result = F.grid_sample(image, grid, align_corners=False)
    torch.testing.assert_close(result, image, atol=2e-5, rtol=0)


def test_common_support_erodes_image_border_and_unsupported_neighbor():
    a = np.ones((12, 16), bool)
    b = a.copy()
    b[5, 5] = False
    shared = eroded_common_support([a, b])
    assert not shared[0].any() and not shared[-1].any()
    assert not shared[:, 0].any() and not shared[:, -1].any()
    assert not shared[4:7, 4:7].any()
    assert shared.sum() == 10 * 14 - 9


def test_gradient_control_polarity_and_misalignment():
    rng = np.random.default_rng(3)
    image = torch.tensor(rng.random((1, 1, 32, 40)), dtype=torch.float32)
    support = eroded_common_support([np.ones((32, 40), bool)])
    same = image_scores(image, image, support)
    inverse = image_scores(image, 1 - image, support)
    shifted = image_scores(image, torch.roll(image, 5, -1), support)
    assert same["signed_ngcc"] > 0.999
    assert inverse["signed_ngcc"] < -0.999 and inverse["absolute_ngcc"] > 0.999
    assert shifted["absolute_ngcc"] < 0.2


def test_flat_and_empty_support_do_not_become_perfect_scores():
    image = torch.ones((1, 1, 32, 40))
    support = eroded_common_support([np.ones((32, 40), bool)])
    assert all(v is None for v in image_scores(image, image, support).values())
    assert all(v is None for v in image_scores(image, image, support & False).values())
