#!/usr/bin/env python3
"""Matched extra-training control for the v5 pilot, retaining the v4 objective.

The v4 initialization, optimizer, learning-rate schedule, frame order, precision,
and loader mode match train_antiuav300_registration_v5.py. Only the objective
changes. This is an additional 10-epoch control, not a new qualified model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from aero_ir.registration.protocol_v5 import INITIAL_CHECKPOINT_SHA256, LEARNING_RATE
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.registration.training_v4 import training_step
from aero_ir.utils.manifest import file_sha256
from scripts.train_antiuav300_registration_v2 import _build_cache, _epoch_arrays, _save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    for key in ("root", "initial_checkpoint", "cache_root", "output_dir"):
        if any(char in str(getattr(args, key)) for char in "\n\r"):
            parser.error(f"{key} contains a newline; use a single-line path")
    if not args.initial_checkpoint.is_file():
        parser.error(f"checkpoint is missing: {args.initial_checkpoint}")
    if file_sha256(args.initial_checkpoint) != INITIAL_CHECKPOINT_SHA256:
        parser.error("control requires the same completed v4 checkpoint as the v5 pilot")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("refusing to overwrite a non-empty control output directory")
    if not (args.cache_root / "manifest.json").is_file():
        parser.error("control requires the existing complete cache; no cache is built")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cache = _build_cache(args.root, args.cache_root)
    if cache.get("fit_split") != "train" or cache.get("validation_or_test_access") != "none":
        parser.error("control requires a train-only cache")
    if len(cache["shards"]) * 16 % args.batch_size:
        parser.error("cache size is not divisible by batch size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_superfusion_matcher(args.initial_checkpoint, device)
    # Preserve the loader's eval mode to match the already-running v5 pilot.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    steps_per_epoch = len(cache["shards"]) * 16 // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-6
    )
    source_paths = [
        Path(__file__),
        Path("src/aero_ir/registration/protocol_v4.py"),
        Path("src/aero_ir/registration/training_v4.py"),
        Path("src/aero_ir/registration/protocol_v5.py"),
        Path("scripts/train_antiuav300_registration_v5.py"),
        Path("scripts/train_antiuav300_registration_v2.py"),
    ]
    project = Path(__file__).resolve().parents[1]
    sources = {str(p): file_sha256(project / p) for p in source_paths}
    metadata = {
        "schema_version": 5,
        "kind": "antiuav300_train_only_registration_v5_matched_v4_control",
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "initial_checkpoint_sha256": file_sha256(args.initial_checkpoint),
        "cache_manifest_sha256": cache["cache_manifest_sha256"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "seed": args.seed,
        "objective": "v4_bidirectional_geometry_continued_control",
        "model_training_mode": model.training,
        "source_sha256": sources,
        "optimizer_steps": args.epochs * steps_per_epoch,
        "generator_training_eligible": "hold",
    }
    (args.output_dir / "run_spec.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for epoch in range(args.epochs):
        arrays = _epoch_arrays(args.cache_root, cache["shards"], epoch, args.seed)
        totals: dict[str, float] = {}
        steps = 0
        for offset in range(0, len(arrays[0]), args.batch_size):
            sl = slice(offset, offset + args.batch_size)
            visible = torch.from_numpy(arrays[0][sl]).permute(0, 3, 1, 2).to(device).float() / 255
            infrared = torch.from_numpy(arrays[1][sl]).permute(0, 3, 1, 2).to(device).float() / 255
            visible_boxes = torch.from_numpy(arrays[2][sl]).to(device)
            infrared_boxes = torch.from_numpy(arrays[3][sl]).to(device)
            values, _ = training_step(
                model, optimizer, visible, infrared, visible_boxes, infrared_boxes
            )
            scheduler.step()
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        record = {"epoch": epoch + 1, "loss": {k: v / steps for k, v in sorted(totals.items())}}
        with (args.output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
    if sources != {str(p): file_sha256(project / p) for p in source_paths}:
        raise RuntimeError("source changed during control; refusing mixed-provenance checkpoint")
    final = args.output_dir / f"antiuav300_registration_v5_control_e{args.epochs}.pth"
    _save_checkpoint(final, model, optimizer, scheduler, args.epochs, metadata)
    print(json.dumps({"checkpoint": str(final), "checkpoint_sha256": file_sha256(final)}))


if __name__ == "__main__":
    main()
