import random
from pathlib import Path

import numpy as np
import pytest
import torch

from aero_ir.detect.yolox_flir_exp import (
    Exp,
    deterministic_worker_init,
    enforce_deterministic_torch,
)
from aero_ir.detect.yolox_trainer import (
    AccumulatingTrainer,
    accumulation_step,
    summarize_timing_records,
)


def test_accumulation_schedule_handles_complete_and_partial_groups():
    steps = [accumulation_step(index, iterations_per_epoch=10, steps=4) for index in range(10)]

    assert [step.group_size for step in steps] == [4] * 8 + [2] * 2
    assert [index for index, step in enumerate(steps) if step.starts_group] == [0, 4, 8]
    assert [index for index, step in enumerate(steps) if step.ends_group] == [3, 7, 9]
    assert [step.optimizer_step_in_epoch for step in steps] == [1] * 4 + [2] * 4 + [3] * 2


@pytest.mark.parametrize(
    ("iteration", "iterations_per_epoch", "steps"),
    [(-1, 10, 4), (10, 10, 4), (0, 0, 4), (0, 10, 0)],
)
def test_accumulation_schedule_rejects_invalid_values(iteration, iterations_per_epoch, steps):
    with pytest.raises(ValueError):
        accumulation_step(iteration, iterations_per_epoch, steps)


def test_flir_exp_scales_optimizer_and_scheduler_by_effective_batch(monkeypatch):
    monkeypatch.setenv("AERO_FLIR_ROOT", str(Path.cwd()))
    monkeypatch.setenv("AERO_YOLOX_MAX_EPOCHS", "1")
    monkeypatch.setenv("AERO_YOLOX_GRAD_ACCUM", "8")
    monkeypatch.setenv("AERO_YOLOX_EFFECTIVE_BATCH", "64")
    exp = Exp()

    exp.get_model()
    optimizer = exp.get_optimizer(batch_size=8)
    scheduler = exp.get_lr_scheduler(lr=-1.0, iters_per_epoch=16)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert scheduler.iters_per_epoch == 2
    assert scheduler.total_iters == 2
    assert scheduler.lr == pytest.approx(0.01)


def test_trainer_rejects_a_mismatched_effective_batch():
    exp = type("FixtureExp", (), {"gradient_accumulation_steps": 8, "effective_batch_size": 64})()
    args = type("FixtureArgs", (), {"batch_size": 4})()

    with pytest.raises(ValueError, match="4 x accumulation 8 = 32, expected 64"):
        AccumulatingTrainer(exp, args)


def test_flir_exp_rejects_newline_in_dataset_root(monkeypatch):
    monkeypatch.setenv("AERO_FLIR_ROOT", "/data/extracted/\n  FLIR_ADAS_v2")

    with pytest.raises(ValueError, match="contains a newline"):
        Exp()


def test_worker_augmentation_seed_is_repeatable_and_worker_specific():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        samples = []
        for worker_id in (0, 1, 0):
            deterministic_worker_init(worker_id, base_seed=19)
            samples.append((random.random(), float(np.random.random()), float(torch.rand(1))))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)

    assert samples[0] == samples[2]
    assert samples[0] != samples[1]


def test_torch_determinism_policy_disables_algorithm_benchmarking():
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_deterministic = torch.backends.cudnn.deterministic
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        enforce_deterministic_torch()

        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.use_deterministic_algorithms(previous_algorithms)


def test_timing_summary_excludes_warmup_and_reports_throughput():
    records = [
        {"iteration_time_s": 10.0, "data_time_s": 1.0},
        {"iteration_time_s": 0.5, "data_time_s": 0.1},
        {"iteration_time_s": 1.0, "data_time_s": 0.2},
    ]

    result = summarize_timing_records(records, warmup_iters=1, batch_size=8)

    assert result["measured_iterations"] == 2
    assert result["iteration_time_s"]["mean"] == pytest.approx(0.75)
    assert result["measured_wall_time_s"] == pytest.approx(1.5)
    assert result["network_images_per_s"] == pytest.approx(16 / 1.5)


def test_timing_summary_rejects_nonfinite_measurements():
    with pytest.raises(ValueError, match="positive and finite"):
        summarize_timing_records(
            [{"iteration_time_s": float("nan"), "data_time_s": 0.0}],
            warmup_iters=0,
            batch_size=8,
        )
