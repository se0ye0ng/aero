import torch

from scripts.audit_antiuav300_dense_registration import (
    CHECKPOINT_SHA256,
    _balanced_indices,
    _checkpoint_provenance,
    _gate_report,
    _split_report,
)
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
