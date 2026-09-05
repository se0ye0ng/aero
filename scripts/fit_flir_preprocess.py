#!/usr/bin/env python3
"""Fit the detector's fixed intensity transform from FLIR training data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.preprocess import fit_uint16_linear_preprocess
from aero_ir.data.registry import FLIRThermalDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/flir_trainval_manifest.json"),
    )
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lower-quantile", type=float, default=0.005)
    parser.add_argument("--upper-quantile", type=float, default=0.995)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/flir_preprocess.json"),
    )
    args = parser.parse_args()

    dataset = FLIRThermalDataset(args.root, args.manifest, "train")
    specification = fit_uint16_linear_preprocess(
        dataset,
        manifest_sha256=dataset.manifest["manifest_sha256"],
        sample_size=args.sample_size,
        seed=args.seed,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(specification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(specification, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
