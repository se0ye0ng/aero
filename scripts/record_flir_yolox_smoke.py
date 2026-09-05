#!/usr/bin/env python3
"""Validate and record a completed FLIR YOLOX engineering smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.detect.yolox_smoke import build_yolox_smoke_report, verify_yolox_smoke_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/flir_trainval_manifest.json"),
    )
    parser.add_argument(
        "--preprocess",
        type=Path,
        default=Path("experiments/flir_preprocess.json"),
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("experiments/flir_yolox/preflight.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/flir_yolox_smoke_result.json"),
    )
    args = parser.parse_args()

    report = build_yolox_smoke_report(
        args.run_dir,
        dataset_manifest=args.manifest,
        preprocess=args.preprocess,
        prepared_preflight=args.preflight,
    )
    verify_yolox_smoke_report(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
