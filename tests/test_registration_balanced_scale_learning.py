import pytest
import torch

from scripts.probe_registration_balanced_scale_learning import (
    measured_scale_term,
    panels,
    schedule,
)


def test_panels_cover_all_sequences_and_probes_are_distinct():
    shards = [{"sequence_id": str(i), "pairs": 20 + i} for i in range(8)]
    screen = {
        "rows": [{"sequence_id": s["sequence_id"]} for s in shards],
        "samples": [
            {"selection": [[s["sequence_id"], s["pairs"] // 2]], "array_sha256": "x"}
            for s in shards
        ],
    }
    fit, probe = panels(screen, shards)
    assert {s["sequence_id"] for s in fit} == {s["sequence_id"] for s in probe}
    assert all(a["selection"] != b["selection"] for a, b in zip(fit, probe, strict=True))
    assert all(s["role"] == "fit" for s in fit)
    assert all(s["role"] == "probe" for s in probe)
    with pytest.raises(ValueError, match="every train"):
        panels(screen, shards[:-1])
    screen["samples"][0]["selection"][0][1] = 0
    with pytest.raises(ValueError, match="midpoint"):
        panels(screen, shards)


def test_schedule_visits160_once_and_rejects_probe_input():
    samples = [{"sequence_id": str(i), "role": "fit"} for i in range(160)]
    batches = schedule(samples)
    assert batches == schedule(samples)
    assert batches != schedule(samples, seed=1)
    assert len(batches) == 40
    assert sorted(s["sequence_id"] for b in batches for s in b["samples"]) == sorted(
        s["sequence_id"] for s in samples
    )
    assert all(len(b["samples"]) == 4 for b in batches)
    with pytest.raises(ValueError, match="fit-only"):
        schedule([{**s, "role": "probe"} for s in samples])
    with pytest.raises(ValueError, match="unique"):
        schedule(samples[:-1] + [samples[0]])


def test_unmeasurable_auxiliary_is_explicit_not_a_zero_error_measurement():
    ref = torch.zeros(2, 2, 32, 64)
    ref[1] = 10
    prediction = torch.zeros_like(ref, requires_grad=True)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]] * 2)
    centres = torch.zeros(2, 2)
    scales = torch.full((2, 2), 1.25)
    effective, meta = measured_scale_term(ref, prediction, boxes, centres, scales)
    assert meta["eligible_samples"] == [True, False]
    assert meta["target_pixels"][1] == 0
    assert float(effective.detach()) == pytest.approx(meta["eligible_mean_loss"] / 2)
    effective.backward()
    assert prediction.grad[0].abs().sum() > 0
    assert prediction.grad[1].abs().sum() == 0
    zero, empty = measured_scale_term(ref + 10, prediction, boxes, centres, scales)
    assert zero.item() == 0
    assert empty["eligible_mean_loss"] is None
    assert empty["eligible_samples"] == [False, False]


def test_nonfinite_inputs_not_silently_excluded():
    ref = torch.zeros(1, 2, 32, 32)
    with pytest.raises(ValueError, match="nonfinite"):
        measured_scale_term(
            ref,
            ref * float("nan"),
            torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            torch.zeros(1, 2),
            torch.ones(1, 2),
        )
