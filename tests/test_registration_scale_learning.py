import copy

import pytest
import torch

from aero_ir.registration.geometry import centre_grid
from scripts import probe_registration_scale_learning as probe


class ToyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def fields(self, context, velocity):
        value = self.weight * context + velocity
        return value, -value


def test_selection_is_disjoint_deterministic_and_rejects_duplicate_ids():
    groups = {str(i): [f"{i}b", f"{i}a", f"{i}c"] for i in range(4)}
    chosen = probe.role_selection(groups)
    assert [(role, sid) for _, role, sid in chosen][:2] == [("fit", "0a"), ("probe", "0b")]
    assert len({sid for _, _, sid in chosen}) == 8
    with pytest.raises(ValueError, match="duplicate"):
        probe.role_selection({str(i): ["same", "other"] for i in range(4)})
    with pytest.raises(ValueError, match="two observations"):
        probe.role_selection({"one": ["a", "b"]})


def test_probe_observations_rejected_before_updates():
    head = ToyHead()
    initial = head.weight.detach().clone()
    with pytest.raises(ValueError, match="only fit-role"):
        probe.fit(head, [{"role": "probe"}], weight=0.1)
    assert torch.equal(initial, head.weight)


def test_actual_optimizer_steps_and_matched_schedule(monkeypatch):
    torch.set_num_threads(1)
    zero = torch.zeros(1, 2, 32, 64)
    context = centre_grid(zero).permute(0, 3, 1, 2).clone().requires_grad_()
    sample = {
        "role": "fit",
        "sequence_id": "a",
        "context": context,
        "velocity": zero,
        "centre": torch.zeros(2),
        "visible": torch.zeros(1, 3, 32, 64),
        "infrared": torch.zeros(1, 3, 32, 64),
        "vb": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
        "ib": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
        "variants": [{"context": context, "velocity": zero} for _ in probe.TRAIN_SCALES],
    }

    def geometric(v, ir, vb, ib, fields):
        loss = fields[0].square().mean()
        return loss, {"total": float(loss.detach())}

    monkeypatch.setattr(probe, "training_loss", geometric)
    initial = ToyHead()
    traces = []
    for weight in (0, 0.1):
        head = copy.deepcopy(initial)
        trace, optimizer = probe.fit(head, [sample], weight=weight, steps=6)
        assert len(trace) == 6 and int(optimizer["state"][0]["step"]) == 6
        assert not torch.equal(initial.weight, head.weight)
        assert context.grad is None
        for row in trace:
            expected = (
                row["before_update_geometry"]["total"] + weight * row["before_update_scale_loss"]
            )
            assert row["before_update_total"] == pytest.approx(expected)
        traces.append(trace)
    assert [r["scale_xy"] for r in traces[0]] == [r["scale_xy"] for r in traces[1]]
    assert [r["fit_sequence_ids"] for r in traces[0]] == [r["fit_sequence_ids"] for r in traces[1]]


def test_evaluation_support_cannot_shrink_when_trained_map_leaves_image():
    initial = torch.zeros(1, 2, 32, 64)
    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    centre = torch.zeros(2)
    first = probe.fixed_response(initial, initial, initial, box, centre, 1.25, 1.25)
    bad = probe.fixed_response(initial + 10, initial + 10, initial, box, centre, 1.25, 1.25)
    assert (
        bad["fixed_initial_roi"]["observed_pixels"] == first["fixed_initial_roi"]["observed_pixels"]
    )
    assert bad["fixed_initial_roi"]["current_base_valid_fraction"] == 0
    assert bad["fixed_initial_roi"]["current_prediction_valid_fraction"] == 0
    assert bad["fixed_initial_roi"]["median_error_pixels"] > 10
