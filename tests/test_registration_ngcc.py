import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import centre_grid
from aero_ir.registration.ngcc import METHODS, gradients, scores
from scripts.probe_registration_ngcc import evaluate, fixed_support, synthetic


def fixture():
    torch.manual_seed(11)
    image = torch.rand(1, 1, 64, 64)
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[8:-8, 8:-8] = True
    return image, mask


def test_shift_and_polarity():
    image, _ = fixture()
    result = synthetic(image)
    assert all(result["normal"].values())
    assert result["inverted"]["absolute_ngcc"]
    assert result["inverted"]["ngf_style_squared_cosine"]
    assert not result["inverted"]["signed_ngcc"]


def test_flat_abstention():
    image, mask = fixture()
    losses, valid = scores(torch.ones_like(image), image, mask)
    assert not valid.any()
    assert all(torch.isfinite(x).all() for x in losses.values())


def test_displacement_gradient():
    image, mask = fixture()
    shift = torch.tensor(0.003, requires_grad=True)
    grid = centre_grid(image.new_zeros(1, 2, 64, 64)) + shift
    warped = F.grid_sample(image, grid, align_corners=False)
    losses, valid = scores(warped, image, mask)
    assert valid.all()
    for method in METHODS:
        (derivative,) = torch.autograd.grad(losses[method].sum(), shift, retain_graph=True)
        assert torch.isfinite(derivative) and derivative.abs() > 0


def test_rotation_gradients_in_target_coordinates():
    image, mask = fixture()
    grid = centre_grid(image.new_zeros(1, 2, 64, 64))
    rotation = grid.clone()
    rotation[..., 0], rotation[..., 1] = -grid[..., 1], grid[..., 0]
    warped = F.grid_sample(image, rotation, align_corners=False)
    losses, valid = scores(warped, warped, mask)
    assert valid.all() and losses["signed_ngcc"].abs().max() < 1e-5
    wrong = F.grid_sample(gradients(image), rotation, align_corners=False)
    assert not torch.allclose(wrong[:, :, mask], gradients(warped)[:, :, mask])


def test_header_support_and_empty():
    image, mask = fixture()
    grid = centre_grid(image.new_zeros(1, 2, 64, 64))
    values, counts = evaluate(image, image, grid, {"roi": mask}, 0.9, 0.9)
    assert counts["roi"] == 0
    assert all(not v for v in values["roi"].values())


def test_affine_intensity_invariance():
    image, mask = fixture()
    losses, valid = scores(image, 0.5 * image + 0.2, mask)
    assert valid.all() and losses["signed_ngcc"].abs().max() < 1e-5


def test_mixed_polarity_and_nonfinite():
    image, mask = fixture()
    target = image.clone()
    target[..., :32] = 1 - target[..., :32]
    mask[:, 30:34] = False  # Exclude the artificial contrast-change seam.
    losses, valid = scores(image, target, mask)
    assert valid.all()
    assert losses["ngf_style_squared_cosine"] < losses["absolute_ngcc"]
    broken = image.clone()
    broken[..., 20, 20] = float("nan")
    _, valid = scores(broken, target, mask)
    assert not valid.any()


def test_candidate_permutation():
    image, mask = fixture()
    grid = centre_grid(image.new_zeros(1, 2, 64, 64))
    maps = torch.cat((grid, grid + 0.03, grid - 0.03))
    values, counts = evaluate(image, image, maps, {"roi": mask})
    order = torch.tensor([2, 0, 1])
    other, other_counts = evaluate(image, image, maps[order], {"roi": mask})
    assert counts == other_counts
    for method in METHODS:
        assert torch.allclose(
            torch.tensor(values["roi"][method])[order], torch.tensor(other["roi"][method])
        )


def test_excluded_header_pixels_cannot_change_scores():
    image, mask = fixture()
    grid = centre_grid(image.new_zeros(1, 2, 64, 64))
    # Fractional shift exercises bilinear footprint at the exclusion boundary.
    grid[..., 1] += 0.007
    changed = image.clone()
    changed[..., :13, :] = 0
    original, counts = evaluate(image, image, grid, {"roi": mask}, 0.2, 0.2)
    altered, counts2 = evaluate(changed, changed, grid, {"roi": mask}, 0.2, 0.2)
    assert counts == counts2
    for method in METHODS:
        assert torch.allclose(
            torch.tensor(original["roi"][method]), torch.tensor(altered["roi"][method]), atol=1e-7
        )


def test_all_geometric_validity_is_eroded_for_sobel():
    image, _ = fixture()
    grid = centre_grid(image.new_zeros(1, 2, 64, 64))
    grid[:, 20, 20, 0] = 2  # Isolated out-of-bounds x coordinate.
    support = fixed_support(grid)
    assert not support[19:22, 19:22].any()
    assert support[18, 20]


def test_synthetic_flat_does_not_claim_truth_from_ties():
    result = synthetic(torch.ones(1, 1, 64, 64))
    assert not any(result["normal"].values())
    assert not any(result["inverted"].values())
