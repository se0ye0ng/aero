#!/usr/bin/env python3
"""Prepare or execute one manifest-locked FLIR YOLOX run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.detect.yolox_run import (
    execute_yolox_run,
    prepare_yolox_run_spec,
    save_yolox_run_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--mode", choices=("timing", "train"), required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--root", required=True, type=Path)
    prepare.add_argument("--repository-root", type=Path, default=Path.cwd())
    prepare.add_argument(
        "--manifest", type=Path, default=Path("experiments/flir_trainval_manifest.json")
    )
    prepare.add_argument(
        "--preprocess", type=Path, default=Path("experiments/flir_preprocess.json")
    )
    prepare.add_argument("--prepared-root", type=Path, default=Path("experiments/flir_yolox"))
    prepare.add_argument("--output-root", type=Path, default=Path("experiments/yolox_runs"))
    prepare.add_argument("--spec-out", type=Path)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--epochs", type=int, default=300)
    prepare.add_argument("--batch-size", type=int, default=8)
    prepare.add_argument("--gradient-accumulation-steps", type=int, default=8)
    prepare.add_argument("--effective-batch-size", type=int, default=64)
    prepare.add_argument("--workers", type=int, default=2)
    prepare.add_argument("--max-train-iters", type=int, default=0)
    prepare.add_argument("--timing-warmup-iters", type=int, default=0)
    prepare.add_argument("--print-interval", type=int, default=10)
    prepare.add_argument("--eval-interval", type=int, default=10)
    prepare.add_argument("--no-fp16", action="store_true")

    execute = subparsers.add_parser("execute")
    execute.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "execute":
        manifest_path = execute_yolox_run(args.spec)
        print(f"wrote {manifest_path}")
        return

    spec = prepare_yolox_run_spec(
        mode=args.mode,
        run_id=args.run_id,
        repository_root=args.repository_root,
        dataset_root=args.root,
        dataset_manifest=args.manifest,
        preprocess=args.preprocess,
        prepared_root=args.prepared_root,
        output_root=args.output_root,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=args.effective_batch_size,
        workers=args.workers,
        max_train_iters=args.max_train_iters,
        timing_warmup_iters=args.timing_warmup_iters,
        print_interval=args.print_interval,
        eval_interval=args.eval_interval,
        fp16=not args.no_fp16,
    )
    destination = args.spec_out or Path("experiments/yolox_specs") / f"{args.run_id}.json"
    digest = save_yolox_run_spec(spec, destination)
    print(json.dumps(spec, indent=2, sort_keys=True))
    print(f"wrote {destination} ({digest})")
    print(f"GPU command: .venv/bin/python scripts/run_flir_yolox.py execute --spec {destination}")


if __name__ == "__main__":
    main()
