"""Protocol-preserving gradient accumulation for the pinned YOLOX trainer."""

from __future__ import annotations

import datetime
import json
import math
import os
import socket
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from yolox.core import Trainer
from yolox.utils import gpu_mem_usage


@dataclass(frozen=True)
class AccumulationStep:
    """Position of one microbatch within an optimizer-step group."""

    group_size: int
    starts_group: bool
    ends_group: bool
    optimizer_step_in_epoch: int


def accumulation_step(iteration: int, iterations_per_epoch: int, steps: int) -> AccumulationStep:
    """Describe accumulation, including a shorter final group without loss under-scaling."""
    if iterations_per_epoch <= 0:
        raise ValueError("iterations_per_epoch must be positive")
    if not 0 <= iteration < iterations_per_epoch:
        raise ValueError("iteration is outside the epoch")
    if steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    group_start = (iteration // steps) * steps
    group_end = min(group_start + steps, iterations_per_epoch)
    return AccumulationStep(
        group_size=group_end - group_start,
        starts_group=iteration == group_start,
        ends_group=iteration + 1 == group_end,
        optimizer_step_in_epoch=iteration // steps + 1,
    )


def summarize_timing_records(records: list[dict], warmup_iters: int, batch_size: int) -> dict:
    """Summarize synchronized iteration measurements after a declared warm-up prefix."""
    if warmup_iters < 0 or warmup_iters >= len(records):
        raise ValueError("timing warm-up must leave at least one measured iteration")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    measured = records[warmup_iters:]
    iteration_times = np.asarray([record["iteration_time_s"] for record in measured], dtype=float)
    data_times = np.asarray([record["data_time_s"] for record in measured], dtype=float)
    if not np.isfinite(iteration_times).all() or (iteration_times <= 0).any():
        raise ValueError("iteration timings must be positive and finite")
    if not np.isfinite(data_times).all() or (data_times < 0).any():
        raise ValueError("data timings must be non-negative and finite")
    elapsed = float(iteration_times.sum())
    return {
        "warmup_iterations": warmup_iters,
        "measured_iterations": len(measured),
        "iteration_time_s": {
            "mean": float(iteration_times.mean()),
            "median": float(np.median(iteration_times)),
            "p95": float(np.percentile(iteration_times, 95)),
            "min": float(iteration_times.min()),
            "max": float(iteration_times.max()),
        },
        "data_time_s": {
            "mean": float(data_times.mean()),
            "p95": float(np.percentile(data_times, 95)),
        },
        "measured_wall_time_s": elapsed,
        "network_images_per_s": batch_size * len(measured) / elapsed,
    }


class AccumulatingTrainer(Trainer):
    """YOLOX 0.3.0 trainer with explicit effective-batch accumulation.

    The upstream CLI batch remains the number of images loaded per microbatch. Learning-rate
    scaling, scheduling, EMA updates, and random-resize cadence follow optimizer steps and the
    declared effective batch instead of microbatches.
    """

    def __init__(self, exp, args):
        self.accumulation_steps = int(exp.gradient_accumulation_steps)
        self.effective_batch_size = int(exp.effective_batch_size)
        if self.accumulation_steps <= 0:
            raise ValueError("gradient accumulation steps must be positive")
        actual_effective_batch = int(args.batch_size) * self.accumulation_steps
        if actual_effective_batch != self.effective_batch_size:
            raise ValueError(
                "YOLOX batch mismatch: CLI batch "
                f"{args.batch_size} x accumulation {self.accumulation_steps} = "
                f"{actual_effective_batch}, expected {self.effective_batch_size}"
            )
        self.run_mode = str(exp.run_mode)
        self.max_train_iters = int(exp.max_train_iters)
        self.timing_warmup_iters = int(exp.timing_warmup_iters)
        self._timing_records: list[dict] = []
        self._timing_completed = False
        self._training_completed = False
        super().__init__(exp, args)

    def before_train(self):
        super().before_train()
        self.full_iterations_per_epoch = self.max_iter
        if self.run_mode == "timing":
            if self.max_train_iters > self.full_iterations_per_epoch:
                raise ValueError(
                    f"timing cap {self.max_train_iters} exceeds full epoch "
                    f"{self.full_iterations_per_epoch}"
                )
            if self.max_train_iters % self.accumulation_steps:
                raise ValueError("timing cap must be divisible by gradient accumulation steps")
            if not 0 <= self.timing_warmup_iters < self.max_train_iters:
                raise ValueError("timing warm-up must leave measured iterations")
            self.max_iter = self.max_train_iters
        self.optimizer_steps_per_epoch = math.ceil(
            self.full_iterations_per_epoch / self.accumulation_steps
        )
        self.executed_optimizer_steps = math.ceil(self.max_iter / self.accumulation_steps)
        self.optimizer_step_count = self.start_epoch * self.optimizer_steps_per_epoch
        if self.use_model_ema:
            self.ema_model.updates = self.optimizer_step_count
        torch.cuda.reset_peak_memory_stats(self.device)
        logger.info(
            "Effective batch: {} (microbatch {} x accumulation {}); optimizer steps/epoch: {}",
            self.effective_batch_size,
            self.args.batch_size,
            self.accumulation_steps,
            self.optimizer_steps_per_epoch,
        )
        if self.run_mode == "timing":
            torch.cuda.reset_peak_memory_stats(self.device)
            logger.info(
                "Bounded full-data timing: {} of {} microbatches; warmup: {}; "
                "mosaic: {}; mixup probability: {}",
                self.max_iter,
                self.full_iterations_per_epoch,
                self.timing_warmup_iters,
                self.train_loader.batch_sampler.mosaic,
                self.exp.mixup_prob,
            )

    def train_in_epoch(self):
        if self.run_mode != "timing":
            result = super().train_in_epoch()
            self._training_completed = True
            return result
        self.epoch = self.start_epoch
        self.before_epoch()
        self.train_in_iter()
        self._timing_completed = True
        return None

    def train_one_iter(self):
        if self.run_mode == "timing":
            torch.cuda.synchronize(self.device)
        iter_start_time = time.time()
        step = accumulation_step(self.iter, self.max_iter, self.accumulation_steps)
        if step.starts_group:
            self.optimizer.zero_grad()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        sync_context = (
            self.model.no_sync() if self.is_distributed and not step.ends_group else nullcontext()
        )
        with sync_context, torch.amp.autocast("cuda", enabled=self.amp_training):
            outputs = self.model(inps, targets)
            backward_loss = outputs["total_loss"] / step.group_size
        self.scaler.scale(backward_loss).backward()

        if step.ends_group:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer_step_count += 1
            if self.use_model_ema:
                self.ema_model.update(self.model)
            lr = self.lr_scheduler.update_lr(self.optimizer_step_count)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = float(self.optimizer.param_groups[0]["lr"])

        if self.run_mode == "timing":
            torch.cuda.synchronize(self.device)
        iter_end_time = time.time()
        if self.run_mode == "timing":
            self._timing_records.append(
                {
                    "iteration": self.iter + 1,
                    "optimizer_step": self.optimizer_step_count,
                    "iteration_time_s": iter_end_time - iter_start_time,
                    "data_time_s": data_end_time - iter_start_time,
                    "input_size": list(self.input_size),
                    "total_loss": float(outputs["total_loss"].detach()),
                }
            )
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
        self._completed_optimizer_step = step.ends_group

    def after_iter(self):
        """Keep upstream logging but resize every ten optimizer steps, not microbatches."""
        if (self.iter + 1) % self.exp.print_interval == 0:
            if self.run_mode == "timing":
                left_iters = self.max_iter - (self.iter + 1)
            else:
                left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
            progress_str = (
                f"epoch: {self.epoch + 1}/{self.max_epoch}, iter: {self.iter + 1}/{self.max_iter}"
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(f"{key}: {value.latest:.1f}" for key, value in loss_meter.items())
            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(f"{key}: {value.avg:.3f}s" for key, value in time_meter.items())
            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}, size: {:d}, {}, optimizer_step: {}",
                progress_str,
                gpu_mem_usage(),
                time_str,
                loss_str,
                self.meter["lr"].latest,
                self.input_size[0],
                eta_str,
                self.optimizer_step_count,
            )
            if self.rank == 0 and self.args.logger == "wandb":
                self.wandb_logger.log_metrics(
                    {key: value.latest for key, value in loss_meter.items()}
                )
                self.wandb_logger.log_metrics({"lr": self.meter["lr"].latest})
            self.meter.clear_meters()

        if self._completed_optimizer_step and self.optimizer_step_count % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader,
                self.epoch,
                self.rank,
                self.is_distributed,
            )

    def after_train(self):
        if self.run_mode != "timing":
            if not self._training_completed:
                logger.error("YOLOX training did not complete")
                return None
            result = super().after_train()
            self._write_runtime_record()
            self._close_tensorboard()
            return result
        if not self._timing_completed:
            logger.error("Bounded full-data timing did not complete")
            return None

        timing = summarize_timing_records(
            self._timing_records,
            self.timing_warmup_iters,
            int(self.args.batch_size),
        )
        losses = np.asarray([record["total_loss"] for record in self._timing_records])
        payload = {
            "schema_version": 1,
            "kind": "flir_yolox_bounded_full_data_timing",
            "status": "pass",
            "scientific_status": "engineering_only_not_reportable",
            "full_dataset_images": len(self.train_loader.dataset),
            "full_iterations_per_epoch": self.full_iterations_per_epoch,
            "executed_iterations": self.max_iter,
            "executed_optimizer_steps": self.executed_optimizer_steps,
            "configured_epochs": self.max_epoch,
            "executed_epochs": 1,
            "microbatch_size": int(self.args.batch_size),
            "gradient_accumulation_steps": self.accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "normal_augmentation": {
                "mosaic": bool(self.train_loader.batch_sampler.mosaic),
                "mosaic_probability": float(self.exp.mosaic_prob),
                "mixup": bool(self.exp.enable_mixup),
                "mixup_probability": float(self.exp.mixup_prob),
                "multiscale_range": int(self.exp.multiscale_range),
            },
            "input_sizes_seen": sorted(
                {tuple(record["input_size"]) for record in self._timing_records}
            ),
            "all_losses_finite": bool(np.isfinite(losses).all()),
            "last_total_loss": float(losses[-1]),
            "timing": timing,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(self.device) / (1024**2),
            "runtime": {
                "host": socket.gethostname(),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(self.device),
            },
        }
        if not payload["all_losses_finite"]:
            raise RuntimeError("bounded timing produced a non-finite loss")
        from aero_ir.utils.manifest import canonical_hash

        payload["metrics_sha256"] = canonical_hash(payload)
        path = Path(self.file_name) / "timing_metrics.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._write_runtime_record()
        self._close_tensorboard()
        logger.info(
            "Bounded full-data timing is done; metrics: {}; hash: {}",
            path,
            payload["metrics_sha256"],
        )
        return None

    def _write_runtime_record(self):
        from aero_ir.utils.manifest import canonical_hash

        payload = {
            "schema_version": 1,
            "status": "pass",
            "host": socket.gethostname(),
            "python_version": __import__("sys").version.split()[0],
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(self.device),
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(self.device) / (1024**2),
            "determinism": {
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "torch_deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
            },
        }
        payload["runtime_sha256"] = canonical_hash(payload)
        path = Path(self.file_name) / "runtime.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _close_tensorboard(self):
        if self.rank == 0 and self.args.logger == "tensorboard":
            self.tblogger.flush()
            self.tblogger.close()
