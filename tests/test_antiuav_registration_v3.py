import copy

import torch

from aero_ir.registration.protocol_v3 import (
    BATCH_SIZE,
    EPOCHS,
    bidirectional_geometry_loss,
    inverse_consistency_statistics,
)
from scripts.audit_antiuav300_bidirectional_registration import gate_report


def test_reciprocal_translation_has_zero_inverse_consistency_error():
    forward = torch.zeros(2, 2, 32, 32)
    forward[:, 0] = 0.10
    forward[:, 1] = -0.06
    reverse = -forward

    statistics = inverse_consistency_statistics(forward, reverse)

    assert statistics["target_cycle_max"].max() < 1e-6
    assert statistics["source_cycle_max"].max() < 1e-6
    assert (statistics["target_cycle_valid_fraction"] > 0.8).all()
    assert (statistics["source_cycle_valid_fraction"] > 0.8).all()


def test_bidirectional_loss_is_finite_and_backpropagates_both_fields():
    visible = torch.rand(2, 3, 256, 256)
    infrared = visible.clone()
    boxes = torch.tensor([[0.5, 0.5, 0.08, 0.06], [0.3, 0.7, 0.04, 0.04]])
    forward = torch.zeros(2, 2, 256, 256, requires_grad=True)
    reverse = torch.zeros(2, 2, 256, 256, requires_grad=True)

    loss, telemetry = bidirectional_geometry_loss(visible, infrared, boxes, boxes, forward, reverse)
    loss.backward()

    assert EPOCHS == 300
    assert BATCH_SIZE == 8
    assert torch.isfinite(loss)
    assert telemetry["inverse_consistency"] < 1e-6
    assert forward.grad is not None and torch.isfinite(forward.grad).all()
    assert reverse.grad is not None and torch.isfinite(reverse.grad).all()


def _passing_split() -> dict:
    summary = {"p50": 0.03}
    direction = {"frame_pass_rate": {"joint": 0.96}}
    return {
        "visible_to_infrared_backward_map": copy.deepcopy(direction),
        "infrared_to_visible_backward_map": copy.deepcopy(direction),
        "forward_edge_ncc_gain": copy.deepcopy(summary),
        "reverse_edge_ncc_gain": copy.deepcopy(summary),
        "frame_pass_rate": {
            "forward_valid_flow": 0.96,
            "reverse_valid_flow": 0.96,
            "forward_positive_jacobian": 0.96,
            "reverse_positive_jacobian": 0.96,
            "target_cycle": 0.96,
            "source_cycle": 0.96,
            "target_cycle_valid": 0.96,
            "source_cycle_valid": 0.96,
            "forward_edge_ncc_improved": 0.8,
            "reverse_edge_ncc_improved": 0.8,
        },
    }


def test_v3_gate_requires_both_native_directions_and_full_audit():
    metrics = {"train": _passing_split(), "val": _passing_split()}
    screen = gate_report(metrics, exhaustive=False)
    assert screen["sequence_balanced_screen"] == "pass"
    assert screen["generator_training_eligible"] == "hold_pending_exhaustive_audit"

    full = gate_report(metrics, exhaustive=True)
    assert full["generator_training_eligible"] == "pass"

    metrics["val"]["infrared_to_visible_backward_map"]["frame_pass_rate"]["joint"] = 0.94
    assert gate_report(metrics, exhaustive=True)["generator_training_eligible"] == "hold"
