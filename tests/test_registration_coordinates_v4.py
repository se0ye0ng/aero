"""Analytic geometry and fail-closed qualification; no GPU or dataset required."""

import copy
import json
import sys

import numpy as np
import pytest
import torch

from aero_ir.registration.geometry import (
    box_points,
    centre_grid,
    cycle_at,
    cycle_field,
    from_superfusion,
    jacobian_determinant,
    map_boxes,
    map_points,
    pixel_norm,
    sampling_map,
    warp,
)
from aero_ir.registration.protocol_v4 import bidirectional_geometry_loss
from aero_ir.registration.qualification_v4 import (
    direction_pass,
    direction_statistics,
    gate_report,
    split_report,
)
from aero_ir.registration.superfusion import DenseMatcher, warp_source_to_target
from scripts.audit_antiuav300_dense_registration import FramePair
from scripts.audit_antiuav300_registration_v4 import checkpoint_provenance
from scripts.train_antiuav300_registration_v4 import EPOCHS, SCHEMA_VERSION


def _affine(matrix, translation=(0.0, 0.0), height=48, width=64):
    field = torch.zeros(1, 2, height, width, dtype=torch.float64)
    grid = centre_grid(field)
    matrix = torch.tensor(matrix, dtype=field.dtype)
    translation = torch.tensor(translation, dtype=field.dtype)
    mapped = grid @ matrix.T + translation
    return (mapped - grid).permute(0, 3, 1, 2)


@pytest.mark.parametrize("shape", [(32, 32), (27, 43)])
def test_identity_image_points_boxes_cycles_and_jacobian_agree(shape):
    h, w = shape
    zero = torch.zeros(1, 2, h, w, dtype=torch.float64)
    image = torch.rand(1, 3, h, w, dtype=torch.float64)
    boxes = torch.tensor([[0.7, 0.6, 0.10, 0.08]], dtype=torch.float64)
    assert torch.allclose(warp(image, zero), image, atol=1e-12)
    assert torch.allclose(map_boxes(boxes, zero), boxes, atol=1e-12)
    points = box_points(boxes)
    assert torch.equal(map_points(zero, points), points)
    residual, valid = cycle_field(zero, zero)
    assert valid.all() and residual.abs().max() == 0
    assert torch.allclose(
        jacobian_determinant(zero), torch.ones(1, h - 1, w - 1, dtype=torch.float64)
    )


def test_adapter_preserves_legacy_image_warp_but_corrects_raw_zero_semantics():
    torch.manual_seed(12)
    raw = torch.randn(1, 2, 31, 47, dtype=torch.float64) * 0.02
    image = torch.rand(1, 3, 31, 47, dtype=torch.float64)
    assert torch.allclose(
        warp(image, from_superfusion(raw)), warp_source_to_target(image, raw), atol=1e-12
    )
    # Raw zero means endpoint sampling, NOT identity. The corrected cycle sees it.
    raw.zero_()
    corrected = from_superfusion(raw)
    residual, valid = cycle_field(corrected, corrected)
    assert pixel_norm(residual[valid], 31, 47).max() > 0.5
    assert not torch.allclose(warp(image, corrected), image)
    # The raw field that actually encodes identity cancels the adapter's base shift.
    assert from_superfusion(-corrected).abs().max() < 1e-12


@pytest.mark.parametrize(
    "matrix,translation",
    [
        ([[1.0, 0], [0, 1.0]], (0.12, -0.08)),
        ([[1.4, 0], [0, 0.8]], (0.0, 0.0)),
        ([[0.8, -0.6], [0.6, 0.8]], (0.0, 0.0)),
        ([[1.0, 0.3], [-0.2, 1.0]], (0.04, 0.03)),
    ],
)
def test_affine_points_warp_box_and_reciprocal_composition(matrix, translation):
    forward = _affine(matrix, translation)
    a = torch.tensor(matrix, dtype=forward.dtype)
    t = torch.tensor(translation, dtype=forward.dtype)
    inv = torch.linalg.inv(a)
    reverse = _affine(inv.tolist(), (-inv @ t).tolist())
    points = torch.tensor([[[-0.2, 0.1], [0.3, -0.2]]], dtype=forward.dtype)
    assert torch.allclose(map_points(forward, points), points @ a.T + t, atol=1e-12)
    residual, valid = cycle_at(forward, reverse, points)
    assert valid.all() and residual.abs().max() < 1e-12
    # A coordinate-channel image directly measures where the image warp samples.
    coordinates = centre_grid(forward).permute(0, 3, 1, 2)
    warped = warp(coordinates, forward).permute(0, 2, 3, 1)
    mapped = sampling_map(forward)
    interior = (mapped.abs() < 0.8).all(dim=-1)
    assert torch.allclose(warped[interior], mapped[interior], atol=1e-12)
    box = torch.tensor([[0.5, 0.5, 0.1, 0.12]], dtype=forward.dtype)
    mapped_support = (box_points(box) @ a.T + t + 1) / 2
    lo, hi = mapped_support.amin(1), mapped_support.amax(1)
    expected = torch.cat(((lo + hi) / 2, hi - lo), dim=1)
    assert torch.allclose(map_boxes(box, forward), expected, atol=1e-12)
    assert torch.allclose(
        jacobian_determinant(forward),
        torch.full((1, 47, 63), torch.linalg.det(a), dtype=forward.dtype),
    )


def _identity_row():
    zero = torch.zeros(1, 2, 64, 64)
    box = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    return direction_statistics(zero, zero, box, box)[0]


def test_identity_passes_engineering_but_never_clears_independent_evidence_hold():
    row = _identity_row()
    assert direction_pass(row)
    rows = [{"sequence_id": "one", "ir_to_rgb_points": row, "rgb_to_ir_points": row}]
    metrics = {split: split_report(rows) for split in ("train", "val")}
    gates = gate_report(metrics, exhaustive=True, complete_coverage=True)
    assert gates["exhaustive_geometry"] == "pass"
    assert gates["independent_correspondence"] == "hold_not_evaluated"
    assert gates["generator_training_eligible"] == "hold"
    assert (
        gate_report(metrics, exhaustive=True, complete_coverage=False)["geometric_screen"] == "hold"
    )


def test_local_error_cannot_hide_in_whole_image_mean():
    first = torch.zeros(1, 2, 64, 64)
    second = first.clone()
    first[:, 0, 29:35, 29:35] = 0.10
    box = torch.tensor([[0.5, 0.5, 0.05, 0.05]])
    residual, _ = cycle_field(first, second)
    assert torch.linalg.vector_norm(residual, dim=-1).mean() < 0.01  # old gate's limit
    row = direction_statistics(first, second, box, box)[0]
    assert row["roi_cycle_max_pixels"] > 2
    assert not direction_pass(row)


@pytest.mark.parametrize("failure", ["empty", "nan", "fold"])
def test_no_support_nonfinite_and_fold_fail_closed(failure):
    first = torch.zeros(1, 2, 32, 32)
    second = first.clone()
    if failure == "empty":
        first.fill_(5)
    elif failure == "nan":
        first[:, :, 15, 15] = float("nan")
    else:
        first = _affine([[-1.0, 0], [0, 1.0]], height=32, width=32).float()
        second = first.clone()
    box = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    row = direction_statistics(first, second, box, box)[0]
    assert not direction_pass(row)
    json.dumps(row, allow_nan=False)
    if failure == "empty":
        assert row["cycle_p95_pixels"] is None


def test_nan_and_missing_metric_are_not_passing_values():
    for value in (None, float("nan"), float("inf")):
        row = _identity_row()
        row["roi_cycle_max_pixels"] = value
        assert not direction_pass(row)


def test_joint_gate_does_not_combine_disjoint_passing_frame_sets():
    good = _identity_row()
    bad = copy.deepcopy(good)
    bad["box_in_bounds"] = False
    rows = [
        {
            "sequence_id": "one",
            "ir_to_rgb_points": bad if i < 5 else good,
            "rgb_to_ir_points": bad if 5 <= i < 10 else good,
        }
        for i in range(100)
    ]
    report = split_report(rows)
    assert report["joint_frame_pass_rate"] == 0.9  # each direction alone is 95%
    metrics = {split: report for split in ("train", "val")}
    assert (
        gate_report(metrics, exhaustive=True, complete_coverage=True)["geometric_screen"] == "hold"
    )


def test_corrected_loss_is_finite_and_backpropagates_into_raw_fields():
    assert EPOCHS == 300 and SCHEMA_VERSION == 4
    torch.manual_seed(2)
    visible = torch.rand(1, 3, 32, 48)
    infrared = visible.clone()
    zero = torch.zeros(1, 2, 32, 48)
    raw_identity = -from_superfusion(zero)
    forward = raw_identity.clone().requires_grad_()
    reverse = raw_identity.clone().requires_grad_()
    boxes = torch.tensor([[0.5, 0.5, 0.15, 0.12]])
    loss, telemetry = bidirectional_geometry_loss(visible, infrared, boxes, boxes, forward, reverse)
    loss.backward()
    assert torch.isfinite(loss)
    assert "worst_perimeter_fraction" not in telemetry
    assert telemetry["inverse_consistency"] < 1e-6
    for field in (forward, reverse):
        assert field.grad is not None and torch.isfinite(field.grad).all()


def test_epoch_metadata_alone_cannot_qualify_an_intermediate_checkpoint(tmp_path):
    path = tmp_path / "intermediate.pth"
    metadata = {
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "epochs": 300,
    }
    torch.save({"DM": {}, "aero_registration": metadata, "epoch": 10}, path)
    with pytest.raises(ValueError, match="actually completed"):
        checkpoint_provenance(path)


def test_uniform_images_do_not_produce_nonfinite_loss_gradients():
    visible = torch.zeros(1, 3, 16, 24)
    forward = torch.zeros(1, 2, 16, 24, requires_grad=True)
    reverse = torch.zeros_like(forward, requires_grad=True)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    loss, _ = bidirectional_geometry_loss(visible, visible, boxes, boxes, forward, reverse)
    loss.backward()
    assert torch.isfinite(forward.grad).all()
    assert torch.isfinite(reverse.grad).all()


def test_real_matcher_v4_loss_forward_backward_on_cpu():
    torch.manual_seed(3)
    model = DenseMatcher().train()
    visible, infrared = torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)
    boxes = torch.tensor([[0.5, 0.5, 0.06, 0.06]])
    forward = model(infrared, visible, direction="visible_to_infrared")
    reverse = model(infrared, visible, direction="infrared_to_visible")
    loss, _ = bidirectional_geometry_loss(visible, infrared, boxes, boxes, forward, reverse)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("missing_pair", [False, True])
def test_audit_cli_writes_hashed_fail_closed_report_without_gpu(
    tmp_path, monkeypatch, missing_pair
):
    import scripts.audit_antiuav300_registration_v4 as audit
    from aero_ir.utils.manifest import canonical_hash

    checkpoint = tmp_path / "complete.pth"
    torch.save(
        {
            "DM": {},
            "epoch": 300,
            "aero_registration": {
                "dataset": "Anti-UAV300",
                "fit_split": "train",
                "validation_or_test_access": "none",
                "epochs": 300,
            },
        },
        checkpoint,
    )

    class IdentityMatcher:
        def eval(self):
            return self

        def __call__(self, infrared, visible, *, direction):
            b, _, h, w = visible.shape
            return -from_superfusion(torch.zeros(b, 2, h, w))

    frames = [
        FramePair(
            split,
            split + "_sequence",
            0,
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.array([0.5, 0.5, 0.1, 0.1], dtype=np.float32),
            np.array([0.5, 0.5, 0.1, 0.1], dtype=np.float32),
        )
        for split in ("train", "val")
    ]
    expected = {split: {(split + "_sequence", 0)} for split in ("train", "val")}
    if missing_pair:
        expected["val"].add(("val_sequence", 1))
    monkeypatch.setattr(audit, "expected_pairs", lambda root, samples: (expected, {}))
    monkeypatch.setattr(audit, "_iter_pairs", lambda *args: iter(frames))
    monkeypatch.setattr(audit, "load_superfusion_matcher", lambda *args: IdentityMatcher())
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--root",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(output),
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 3  # explicit scientific HOLD, even when geometry passes
    result = json.loads(output.read_text())
    digest = result.pop("audit_sha256")
    assert digest == canonical_hash(result)
    assert result["gates"]["geometric_screen"] == ("hold" if missing_pair else "pass")
    assert result["gates"]["generator_training_eligible"] == "hold"
    # The same command must not overwrite the first report.
    previous = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2 and output.read_bytes() == previous
