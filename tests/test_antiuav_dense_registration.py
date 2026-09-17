import json

import torch

from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import (
    CHECKPOINT_SHA256,
    _balanced_indices,
    _checkpoint_provenance,
    _gate_report,
    _split_report,
)
from scripts.check_antiuav300_registration_report import check_report
from scripts.train_antiuav300_registration import _registration_loss


def _passing_rows():
    rows = []
    for index in range(20):
        rows.append(
            {
                "sequence_id": f"sequence-{index % 2}",
                "frame_index": index,
                "predicted_box": [0.5, 0.5, 0.1, 0.1],
                "target_box": [0.5, 0.5, 0.1, 0.1],
                "inverse_residual": 0.0,
                "valid_flow_fraction": 1.0,
                "edge_ncc_before": 0.10,
                "edge_ncc_after": 0.15,
            }
        )
    return rows


def test_balanced_screen_includes_sequence_endpoints():
    assert _balanced_indices(100, 4).tolist() == [0, 33, 66, 99]
    assert _balanced_indices(3, 8).tolist() == [0, 1, 2]


def test_screen_cannot_clear_exhaustive_registration_gate():
    metrics = {split: _split_report(_passing_rows()) for split in ("train", "val")}

    gates = _gate_report(metrics, exhaustive=False)

    assert gates["sequence_balanced_screen"] == "pass"
    assert gates["held_out_validation"] == "candidate"
    assert gates["dense_paired_image_registration"] == "hold_pending_exhaustive_audit"
    assert gates["generator_training_eligible"] == "hold_pending_exhaustive_audit"


def test_exhaustive_gate_requires_dense_and_box_evidence():
    metrics = {split: _split_report(_passing_rows()) for split in ("train", "val")}
    passing = _gate_report(metrics, exhaustive=True)
    assert passing["dense_paired_image_registration"] == "pass"
    assert passing["generator_training_eligible"] == "pass"

    failing_rows = _passing_rows()
    for row in failing_rows:
        row["edge_ncc_after"] = row["edge_ncc_before"]
    metrics["val"] = _split_report(failing_rows)
    failing = _gate_report(metrics, exhaustive=True)
    assert failing["held_out_validation"] == "pass"
    assert failing["dense_paired_image_registration"] == "hold"
    assert failing["generator_training_eligible"] == "hold"


def test_finetuned_checkpoint_provenance_rejects_validation_access(tmp_path):
    metadata = {
        "kind": "antiuav300_train_only_superfusion_finetune",
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "initial_checkpoint_sha256": CHECKPOINT_SHA256,
        "cache_manifest_sha256": "a" * 64,
        "epochs": 300,
        "precision": "float32",
        "seed": 0,
    }
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"aero_registration": metadata}, checkpoint)

    provenance = _checkpoint_provenance(checkpoint, "b" * 64)
    assert provenance["anti_uav_fitting"] == "official train split only"

    metadata["validation_or_test_access"] = "validation"
    torch.save({"aero_registration": metadata}, checkpoint)
    try:
        _checkpoint_provenance(checkpoint, "c" * 64)
    except ValueError as error:
        assert "violates the frozen protocol" in str(error)
    else:
        raise AssertionError("validation-contaminated checkpoint was accepted")


def test_geometry_first_v2_checkpoint_provenance_is_distinct(tmp_path):
    metadata = {
        "schema_version": 2,
        "kind": "antiuav300_train_only_geometry_first_v2",
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "initial_checkpoint_sha256": (
            "a4c8aafe95c098f8b0803980be520aae12122c95f0ed89bddcb5c89305f2388a"
        ),
        "cache_manifest_sha256": "d" * 64,
        "epochs": 300,
        "pairs_per_sequence_per_epoch": 16,
        "unique_train_pairs": 141816,
        "batch_size": 16,
        "precision": "float32",
        "seed": 0,
    }
    checkpoint = tmp_path / "v2.pth"
    torch.save({"aero_registration": metadata}, checkpoint)

    provenance = _checkpoint_provenance(checkpoint, "e" * 64)

    assert provenance["checkpoint_variant"] == "Anti-UAV300_train_only_geometry_first_v2"
    assert provenance["anti_uav_fitting"] == "official train split only"


def test_registration_training_loss_has_finite_flow_gradients():
    visible = torch.rand(2, 3, 256, 256)
    infrared = visible.clone()
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.7, 0.05, 0.08]])
    displacement = torch.zeros(2, 2, 256, 256, requires_grad=True)

    loss, values = _registration_loss(visible, infrared, boxes, boxes, displacement)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    assert displacement.grad is not None
    assert torch.isfinite(displacement.grad).all()


def _affine_target_to_source_field(
    scale: float, translation: tuple[float, float], size: int = 64
) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack(
        (
            (scale - 1.0) * grid_x + translation[0],
            (scale - 1.0) * grid_y + translation[1],
        )
    ).unsqueeze(0)


def test_known_translation_agrees_across_direct_and_inverse_box_paths():
    from aero_ir.registration.superfusion import (
        transform_boxes_source_to_target_newton,
        transform_boxes_target_to_source,
    )

    target = torch.tensor([[0.42, 0.57, 0.16, 0.10]])
    displacement = _affine_target_to_source_field(1.0, (0.12, -0.08))
    source = target + torch.tensor([[0.06, -0.04, 0.0, 0.0]])

    direct = transform_boxes_target_to_source(target, displacement)
    inverse, residual, determinant = transform_boxes_source_to_target_newton(source, displacement)

    assert torch.allclose(direct, source, atol=2e-4)
    assert torch.allclose(inverse, target, atol=2e-4)
    assert residual.max() < 1e-4
    assert determinant.min() > 0.99


def test_newton_inverse_handles_valid_noncontractive_scale():
    from aero_ir.registration.superfusion import (
        transform_boxes_source_to_target,
        transform_boxes_source_to_target_newton,
        transform_boxes_target_to_source,
    )

    scale = 2.1
    target = torch.tensor([[0.48, 0.52, 0.08, 0.06]])
    displacement = _affine_target_to_source_field(scale, (0.02, -0.04))
    source = transform_boxes_target_to_source(target, displacement)
    legacy, legacy_residual = transform_boxes_source_to_target(source, displacement)
    robust, robust_residual, determinant = transform_boxes_source_to_target_newton(
        source, displacement
    )

    assert legacy_residual.item() > 0.1
    assert not torch.allclose(legacy, target, atol=1e-2)
    assert torch.allclose(robust, target, atol=2e-3)
    assert robust_residual.item() < 1e-3
    assert determinant.min() > 4.0


def test_existing_registration_report_is_verified_and_hold_is_preserved(tmp_path):
    checkpoint = tmp_path / "matcher.pth"
    torch.save({"DM": {}}, checkpoint)
    report = {
        "kind": "antiuav300_superfusion_dense_registration_audit",
        "root": str(tmp_path.resolve()),
        "data_usage": {"frame_selection": "8 endpoint-inclusive usable pairs per sequence"},
        "model": {"checkpoint_sha256": file_sha256(checkpoint)},
        "metrics": {
            "train": {"frame_pass_rate": {"joint": 0.74}},
            "val": {"frame_pass_rate": {"joint": 0.67}},
        },
    }
    report["gates"] = _gate_report(report["metrics"], exhaustive=False)
    report["dense_registration_audit_sha256"] = canonical_hash(report)
    report_path = tmp_path / "screen.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    passed, message = check_report(report_path, checkpoint, tmp_path, "screen")

    assert not passed
    assert "HOLD" in message
    assert "74.00%" in message
    report["metrics"]["val"]["frame_pass_rate"]["joint"] = 0.96
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        check_report(report_path, checkpoint, tmp_path, "screen")
    except ValueError as error:
        assert "content hash mismatch" in str(error)
    else:
        raise AssertionError("modified registration result was accepted")


def test_existing_exhaustive_registration_pass_requires_matching_checkpoint(tmp_path):
    checkpoint = tmp_path / "matcher.pth"
    torch.save({"DM": {}}, checkpoint)
    metrics = {split: _split_report(_passing_rows()) for split in ("train", "val")}
    report = {
        "kind": "antiuav300_superfusion_dense_registration_audit",
        "root": str(tmp_path.resolve()),
        "data_usage": {"frame_selection": "all usable paired target frames"},
        "model": {"checkpoint_sha256": file_sha256(checkpoint)},
        "metrics": metrics,
        "gates": _gate_report(metrics, exhaustive=True),
    }
    report["dense_registration_audit_sha256"] = canonical_hash(report)
    report_path = tmp_path / "full.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    passed, message = check_report(report_path, checkpoint, tmp_path, "full")

    assert passed
    assert "PASS" in message
    torch.save({"DM": {"modified": torch.zeros(1)}}, checkpoint)
    try:
        check_report(report_path, checkpoint, tmp_path, "full")
    except ValueError as error:
        assert "different checkpoint" in str(error)
    else:
        raise AssertionError("registration result accepted a different checkpoint")
