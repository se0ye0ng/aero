#!/usr/bin/env python3
"""Train-only v6 preservation pilot; qualification remains a separate evaluation."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from aero_ir.registration.protocol_v5 import INITIAL_CHECKPOINT_SHA256, LEARNING_RATE
from aero_ir.registration.protocol_v6 import PRESERVATION_TOLERANCES, PRESERVATION_WEIGHTS
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.registration.training_v6 import training_step
from aero_ir.utils.manifest import file_sha256
from scripts.train_antiuav300_registration_v2 import _build_cache, _epoch_arrays, _save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "initial-checkpoint", "cache-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.seed < 0:
        parser.error("epochs/batch-size must be positive and seed nonnegative")
    for name in ("root", "initial_checkpoint", "cache_root", "output_dir"):
        if any(char in str(getattr(args, name)) for char in ("\n", "\r")):
            parser.error(f"{name} contains a newline; use a single-line path")
    if not args.initial_checkpoint.is_file():
        parser.error(f"initial checkpoint does not exist: {args.initial_checkpoint}")
    if file_sha256(args.initial_checkpoint) != INITIAL_CHECKPOINT_SHA256:
        parser.error("v6 requires the completed v4 checkpoint for both student and teacher")
    if args.output_dir.exists():
        parser.error("refusing to overwrite an existing v6 output directory; use a new run ID")
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
        parser.error("v6 accepts a train-only cache")
    if cache.get("split_manifest_sha256") != file_sha256(args.root / "label_new" / "train.json"):
        parser.error("cache training split hash differs from requested dataset")
    if not cache["shards"] or len(cache["shards"]) * 16 % args.batch_size:
        parser.error("cache epoch size must be positive and divisible by batch size")
    device = torch.device(args.device)
    model = load_superfusion_matcher(args.initial_checkpoint, device)
    teacher = load_superfusion_matcher(args.initial_checkpoint, device)
    teacher.requires_grad_(False).eval()
    # v5's loader leaves the matcher in evaluation mode. Keep that same mode
    # for matched comparisons; eval() does not disable student gradients.
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    steps_per_epoch = len(cache["shards"]) * 16 // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-6
    )
    project_root = Path(__file__).resolve().parents[1]
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    sources = [Path(__file__).resolve(),
               project_root / "src/aero_ir/registration/protocol_v6.py",
               project_root / "src/aero_ir/registration/training_v6.py",
               project_root / "src/aero_ir/registration/protocol_v5.py",
               project_root / "src/aero_ir/registration/protocol_v4.py",
               project_root / "src/aero_ir/registration/geometry.py",
               project_root / "scripts/train_antiuav300_registration_v2.py"]
    metadata = {
        "schema_version": 6,
        "kind": "antiuav300_train_only_registration_v6_pilot",
        "dataset": "Anti-UAV300", "fit_split": "train", "validation_or_test_access": "none",
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "teacher_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "cache_manifest_sha256": cache["cache_manifest_sha256"],
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": LEARNING_RATE,
        "seed": args.seed, "steps_per_epoch": steps_per_epoch,
        "pairs_per_sequence_per_epoch": 16, "unique_train_pairs": cache["pairs"],
        "objective": "v5_tail_plus_per_pair_v4_box_preservation_v6",
        "preservation_tolerances": PRESERVATION_TOLERANCES,
        "preservation_weights": PRESERVATION_WEIGHTS,
        "constraint_type": "soft_hinge_not_a_feasibility_guarantee",
        "student_mode": "eval_with_gradients_matching_v5", "teacher_mode": "frozen_eval",
        "git_sha": git_sha,
        "source_sha256": {
            str(path.relative_to(project_root)): file_sha256(path) for path in sources
        },
        "torch_version": str(torch.__version__), "device": str(device),
        "generator_training_eligible": "hold_not_qualified",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log_path = args.output_dir / "train_log.jsonl"
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
                model, teacher, optimizer, visible, infrared, visible_boxes, infrared_boxes
            )
            scheduler.step()
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        record = {"epoch": epoch + 1, "loss": {k: v / steps for k, v in sorted(totals.items())}}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        _save_checkpoint(
            args.output_dir / "latest.pth", model, optimizer, scheduler, epoch + 1, metadata
        )
    final = args.output_dir / f"antiuav300_registration_v6_e{args.epochs}.pth"
    _save_checkpoint(final, model, optimizer, scheduler, args.epochs, metadata)
    result = {
        "checkpoint": str(final),
        "checkpoint_sha256": file_sha256(final),
        "epochs_completed": args.epochs,
        "total_optimizer_steps": args.epochs * steps_per_epoch,
        "generator_training_eligible": "hold_not_qualified",
    }
    (args.output_dir / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
