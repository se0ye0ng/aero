#!/usr/bin/env python3
"""Run a train-only, sequence-disjoint RFS diagnostic on Anti-UAV300 IR video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from aero_ir.data.antiuav import load_ir_sequence_sample, load_split_manifest
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
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("experiments/antiuav300_rfs_pilot.json"))
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")
    sequences = sorted(load_split_manifest(args.root, "train"))
    required = 3 * args.sample_size
    if required > len(sequences):
        raise ValueError(f"need {required} disjoint sequences; train manifest has {len(sequences)}")
    rng = np.random.default_rng(300)
    selected = [sequences[index] for index in rng.choice(len(sequences), required, False)]
    partitions = {
        "train_a": sorted(selected[: args.sample_size]),
        "train_b": sorted(selected[args.sample_size : 2 * args.sample_size]),
        "internal_holdout": sorted(selected[2 * args.sample_size :]),
    }
    samples = {
        name: load_ir_sequence_sample(args.root, values, seed=seed)
        for (name, values), seed in zip(partitions.items(), (301, 302, 303), strict=True)
    }

    train_reference = compute_rfs(samples["train_a"][:2], samples["train_b"][:2], PilotRFSCfg())
    holdout_shift = compute_rfs(
        samples["train_a"][:2], samples["internal_holdout"][:2], PilotRFSCfg()
    )
    result = {
        "pilot": "antiuav300_train_only_rfs_real_data",
        "representation": "8-bit grayscale decoded from infrared.mp4",
        "sample_size_per_set": args.sample_size,
        "selection": "one seeded annotated frame per sequence; all three sets sequence-disjoint",
        "scope": "official train split only; official validation and test were not read as samples",
        "class_conditioned": False,
        "train_reference": report_dict(train_reference),
        "train_to_internal_holdout": report_dict(holdout_shift),
        "sample_ids": {name: sample[2] for name, sample in samples.items()},
        "sequence_overlap": {
            "train_a_train_b": sorted(set(partitions["train_a"]) & set(partitions["train_b"])),
            "train_a_internal_holdout": sorted(
                set(partitions["train_a"]) & set(partitions["internal_holdout"])
            ),
            "train_b_internal_holdout": sorted(
                set(partitions["train_b"]) & set(partitions["internal_holdout"])
            ),
        },
        "interpretation": (
            "Diagnostic only: Anti-UAV300 MP4 is display-referred rather than radiometric; the "
            "current RFS pools one class, the sample is small, and no downstream AP was measured."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
