#!/usr/bin/env python3
"""Run the data-free Phase 0 RFS and differentiable-sensor pilot."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from aero_ir.rfs import compute_rfs
from aero_ir.sensor import AGCQuantise, FixedPatternNoise, IRSensorPipeline, MTFBlur, NETDNoise


class PilotRFSCfg:
    components = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    distance = "wasserstein"
    weights = "uniform"
    annulus_dilation_px = 8
    highpass_sigma_px = 1.5


def make_set(
    n_images: int,
    *,
    delta: float,
    noise: float,
    blur: float = 0.0,
    seed: int,
) -> tuple[list[np.ndarray], list[list[tuple[int, int, int, int]]]]:
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    boxes: list[list[tuple[int, int, int, int]]] = []
    for index in range(n_images):
        image = 100.0 + rng.normal(0.0, noise, (96, 96))
        x = 32 + index % 9
        y = 36 + index % 7
        image[y : y + 10, x : x + 10] += delta
        if blur:
            image = ndimage.gaussian_filter(image, blur)
        images.append(image)
        boxes.append([(x, y, 10, 10)])
    return images, boxes


def json_number(value: float) -> float | str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def report_dict(report) -> dict:
    return {
        "scalar": json_number(report.scalar),
        "per_component": {key: json_number(value) for key, value in report.per_component.items()},
        "worst_finite": [[name, json_number(value)] for name, value in report.worst(3)],
    }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def sensor_pilot(device: torch.device, iterations: int) -> dict:
    torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
        torch.cuda.reset_peak_memory_stats(device)

    pipeline = IRSensorPipeline(
        [
            MTFBlur(cutoff_cycles_per_pixel=0.35),
            NETDNoise(netd_mK=50.0, scene_response_dn_per_K=12.0),
            FixedPatternNoise(seed=12345),
            AGCQuantise(bit_depth=8),
        ]
    ).to(device)
    pipeline.train()
    base = 100.0 + 3.0 * torch.randn(8, 1, 256, 320, device=device)

    gradient_is_finite = False
    output_shape: list[int] = []

    def step() -> None:
        nonlocal gradient_is_finite, output_shape
        image = base.detach().clone().requires_grad_(True)
        output = pipeline(image)
        loss = output.square().mean()
        loss.backward()
        output_shape = list(output.shape)
        gradient_is_finite = bool(image.grad is not None and torch.isfinite(image.grad).all())

    for _ in range(2):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started

    result = {
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
        "input_shape": list(base.shape),
        "output_shape": output_shape,
        "forward_backward_iterations": iterations,
        "mean_forward_backward_ms": 1000.0 * elapsed_s / iterations,
        "gradient_is_finite": gradient_is_finite,
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated(device) / (1024**2)
    if not gradient_is_finite:
        raise RuntimeError("sensor-chain input gradient is missing or non-finite")
    return result


def git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("experiments/phase0_pilot.json"))
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    real = make_set(32, delta=20.0, noise=2.0, seed=0)
    faithful = make_set(32, delta=20.0, noise=2.0, seed=1)
    degraded = make_set(32, delta=6.0, noise=0.2, blur=2.0, seed=2)
    inverted = make_set(32, delta=-20.0, noise=2.0, seed=3)

    faithful_report = compute_rfs(real, faithful, PilotRFSCfg())
    degraded_report = compute_rfs(real, degraded, PilotRFSCfg())
    inverted_report = compute_rfs(real, inverted, PilotRFSCfg())
    if not degraded_report.scalar > faithful_report.scalar:
        raise RuntimeError("RFS pilot did not rank the degraded set below the faithful set")
    if not math.isinf(inverted_report.per_component["R2"]):
        raise RuntimeError("RFS pilot did not flag the polarity inversion")

    device = choose_device(args.device)
    result = {
        "pilot": "phase0_data_free",
        "git_sha": git_sha(),
        "working_tree_note": "git_sha identifies the base commit; inspect git diff for changes",
        "torch_version": torch.__version__,
        "rfs": {
            "faithful": report_dict(faithful_report),
            "degraded": report_dict(degraded_report),
            "polarity_inverted": report_dict(inverted_report),
            "checks": {
                "degraded_scores_worse": True,
                "polarity_inversion_flagged": True,
            },
        },
        "sensor": sensor_pilot(device, args.iterations),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
