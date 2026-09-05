#!/usr/bin/env python3
"""Run a small RFS pilot on local 16-bit FLIR thermal images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from aero_ir.data.flir import load_thermal_sample
from aero_ir.rfs import compute_rfs


class PilotRFSCfg:
    components = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    distance = "wasserstein"
    weights = "uniform"
    annulus_dilation_px = 8
    highpass_sigma_px = 1.5


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument(
        "--representation", choices=("analytics16", "display8"), default="analytics16"
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/flir_rfs_pilot.json"))
    args = parser.parse_args()

    train_a = load_thermal_sample(
        args.root,
        "train",
        n_images=args.sample_size,
        seed=17,
        representation=args.representation,
    )
    train_b = load_thermal_sample(
        args.root,
        "train",
        n_images=args.sample_size,
        seed=23,
        representation=args.representation,
        exclude_image_ids=set(train_a[2]),
    )
    validation = load_thermal_sample(
        args.root,
        "val",
        n_images=args.sample_size,
        seed=29,
        representation=args.representation,
    )

    train_reference = compute_rfs(train_a[:2], train_b[:2], PilotRFSCfg())
    validation_shift = compute_rfs(train_a[:2], validation[:2], PilotRFSCfg())
    result = {
        "pilot": "flir_rfs_real_data",
        "representation": args.representation,
        "sample_size_per_set": args.sample_size,
        "selection": "seeded random images with at least one COCO annotation",
        "class_conditioned": False,
        "train_reference": report_dict(train_reference),
        "train_to_validation": report_dict(validation_shift),
        "sample_image_ids": {
            "train_a": train_a[2],
            "train_b": train_b[2],
            "validation": validation[2],
        },
        "interpretation": (
            "Diagnostic only: the current RFS pools classes, the sample is small, and no "
            "downstream AP was measured."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
