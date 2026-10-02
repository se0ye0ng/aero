import numpy as np
import pytest
import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import sampling_map
from aero_ir.registration.mutual_information import joint_pdf, mutual_information
from scripts import probe_registration_mi
from scripts.probe_registration_mi import (
    masks_for_candidates,
    real_check,
    score_maps,
    synthetic_check,
)
from scripts.probe_registration_mind import candidates


def image():
    return F.avg_pool2d(
        torch.rand(1, 1, 64, 64, generator=torch.Generator().manual_seed(73)), 3, 1, 1
    )


def test_pdf_normalization_symmetry_nonnegative():
    a = torch.linspace(0, 1, 100)
    b = a.square()
    for hard in (False, True):
        p = joint_pdf(a, b, hard=hard)
        assert torch.allclose(p.sum(), torch.tensor(1.0))
        assert torch.allclose(p, joint_pdf(b, a, hard=hard).transpose(-1, -2), atol=1e-7)
        assert mutual_information(a, b, hard=hard) >= -1e-6
        assert torch.allclose(
            mutual_information(a, b, hard=hard), mutual_information(b, a, hard=hard), atol=1e-6
        )


def test_hard_histogram_matches_numpy():
    a = torch.linspace(0, 1, 100, dtype=torch.float64)
    b = a.square()
    counts, _, _ = np.histogram2d(a.numpy(), b.numpy(), bins=16, range=((0, 1), (0, 1)))
    p = counts / counts.sum()
    independent = p.sum(1)[:, None] * p.sum(0)[None]
    reference = (p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(independent, 1e-12)))).sum()
    assert np.allclose(joint_pdf(a, b, hard=True)[0].numpy(), p)
    assert float(mutual_information(a, b, hard=True)) == pytest.approx(reference, abs=1e-12)


def test_shuffled_pair_has_less_information_than_true_pair():
    generator = torch.Generator().manual_seed(123)
    a = torch.rand(4096, generator=generator)
    paired = 1 - a
    shuffled = paired[torch.randperm(len(a), generator=generator)]
    for hard in (False, True):
        assert mutual_information(a, paired, hard=hard) > mutual_information(a, shuffled, hard=hard)


def test_known_shift_and_intensity_transforms():
    result = synthetic_check(image())
    for control in result.values():
        assert control["abstention"] is None
        for method in control["methods"].values():
            assert method["exact_recovery"]


def test_synthetic_selection_uses_identity_not_ground_truth(monkeypatch):
    received = []

    def selector(values, baseline):
        received.append(baseline)
        return baseline

    monkeypatch.setattr(probe_registration_mi, "best_index", selector)
    result = synthetic_check(image())
    assert received == [4] * 6  # offsets(-4,0,4), row-major: identity is index4.
    for control in result.values():
        for method in control["methods"].values():
            assert not method["exact_recovery"]


def test_finite_displacement_gradient_matches_difference():
    x = image().double()
    base = sampling_map(torch.zeros(1, 2, 64, 64, dtype=torch.float64))
    shift = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)

    def score(s):
        displacement = torch.stack((s / 32, s * 0))
        warped = F.grid_sample(x, base + displacement, align_corners=False)
        return -mutual_information(
            warped[:, :, 12:-12, 12:-12].flatten(), x[:, :, 12:-12, 12:-12].flatten()
        ).sum()

    score(shift).backward()
    numerical = (score(shift.detach() + 1e-5) - score(shift.detach() - 1e-5)) / 2e-5
    assert torch.isfinite(shift.grad) and shift.grad.abs() > 1e-6
    assert float(shift.grad) == pytest.approx(float(numerical), rel=1e-3, abs=1e-5)


def test_flat_and_empty_support_abstain():
    x = torch.ones(1, 1, 64, 64) * 0.5
    maps, _ = candidates(
        torch.zeros(1, 2, 64, 64), torch.zeros(1, 1, 1, 2), scales=(1,), offsets=(-1, 0, 1)
    )
    result = score_maps(
        x,
        x,
        maps,
        {
            "all": torch.ones(64, 64, dtype=torch.bool),
            "empty": torch.zeros(64, 64, dtype=torch.bool),
        },
    )
    assert result["original"]["all"]["abstention"].startswith("flat")
    assert result["original"]["empty"]["abstention"] == "insufficient_support"


def test_header_and_candidate_order_support():
    maps, _ = candidates(
        torch.zeros(1, 2, 64, 64), torch.zeros(1, 1, 1, 2), scales=(1,), offsets=(-2, 0, 2)
    )
    masks = masks_for_candidates(maps, 0.2, 0.203125)
    assert torch.equal(
        masks["header_excluded"],
        masks_for_candidates(maps.flip(0), 0.2, 0.203125)["header_excluded"],
    )
    assert not masks["header_excluded"][:13].any()
    assert (((maps[:, masks["header_excluded"], 1] + 1) / 2) >= 0.2).all()
    # Even bilinear footprints must exclude rejected header rows.
    contaminated = torch.zeros(1, 1, 64, 64)
    contaminated[:, :, :13] = 1
    sampled = F.grid_sample(contaminated.expand(len(maps), -1, -1, -1), maps, align_corners=False)
    assert sampled[:, 0, masks["header_excluded"]].abs().max() == 0


def test_source_gt_does_not_select_score():
    x = image()
    box = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    other = torch.tensor([[0.55, 0.5, 0.4, 0.4]])
    field = torch.zeros(1, 2, 64, 64)
    a, b = real_check(x, x, box, box, field), real_check(x, x, other, box, field)
    for condition in a["regions"]:
        for region in a["regions"][condition]:
            aa, bb = a["regions"][condition][region], b["regions"][condition][region]
            assert aa["losses"] == bb["losses"]
            for method in aa["selection"]:
                assert aa["selection"][method]["candidate"] == bb["selection"][method]["candidate"]


@pytest.mark.parametrize(
    "a,b",
    [
        (torch.empty(0), torch.empty(0)),
        (torch.tensor([float("nan")]), torch.ones(1)),
        (torch.ones(3), torch.ones(4)),
        (torch.tensor([1.1]), torch.ones(1)),
    ],
)
def test_bad_inputs_rejected(a, b):
    with pytest.raises(ValueError):
        mutual_information(a, b)
