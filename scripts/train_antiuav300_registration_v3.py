#!/usr/bin/env python3
"""Train frozen bidirectional Anti-UAV300 registration protocol v3."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from aero_ir.registration.protocol_v3 import (
    BATCH_SIZE,
    EPOCHS,
    INITIAL_CHECKPOINT_SHA256,
    LEARNING_RATE,
    LOSS_WEIGHTS,
    PAIRS_PER_SEQUENCE_PER_EPOCH,
    bidirectional_geometry_loss,
)
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.train_antiuav300_registration_v2 import (
    _build_cache,
    _epoch_arrays,
    _save_checkpoint,
)

SCHEMA_VERSION = 3


def _git_sha(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _train(args: argparse.Namespace, cache_manifest: dict, project_root: Path) -> Path:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = load_superfusion_matcher(args.initial_checkpoint, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    pairs_per_epoch = len(cache_manifest["shards"]) * PAIRS_PER_SEQUENCE_PER_EPOCH
    if pairs_per_epoch % BATCH_SIZE:
        raise ValueError("frozen v3 epoch size must be divisible by its batch size")
    steps_per_epoch = pairs_per_epoch // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * steps_per_epoch, eta_min=1e-6
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_train_only_bidirectional_v3",
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "initial_checkpoint_sha256": file_sha256(args.initial_checkpoint),
        "cache_manifest_sha256": cache_manifest["cache_manifest_sha256"],
        "git_sha": _git_sha(project_root),
        "epochs": EPOCHS,
        "pairs_per_sequence_per_epoch": PAIRS_PER_SEQUENCE_PER_EPOCH,
        "unique_train_pairs": cache_manifest["pairs"],
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": EPOCHS * steps_per_epoch,
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate": 1e-6,
        "optimizer": "AdamW",
        "loss_weights": LOSS_WEIGHTS,
        "directions": ["visible_to_infrared", "infrared_to_visible"],
        "box_semantics": (
            "score each backward sampling map in its native target-to-source direction"
        ),
        "inverse_consistency": "compose native reciprocal dense maps at off-grid coordinates",
        "grid_convention": (
            "upstream SuperFusion endpoint grid with align_corners=False, retained for "
            "checkpoint compatibility"
        ),
        "precision": "float32",
        "seed": args.seed,
        "selection": "deterministic rotating per-sequence permutations covering every train pair",
        "determinism": (
            "fixed seeds, deterministic cuDNN, TF32 disabled, grid-sample warnings enabled"
        ),
    }
    resume_path = args.output_dir / "latest.pth"
    start_epoch = 0
    if args.resume and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload.get("aero_registration") != metadata:
            raise ValueError("v3 resume checkpoint metadata does not match")
        model.load_state_dict(payload["DM"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"])
        print(f"resuming v3 after epoch {start_epoch}", flush=True)
    elif args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v3 training directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS):
        epoch_data = _epoch_arrays(args.cache_root, cache_manifest["shards"], epoch, args.seed)
        totals: dict[str, float] = {}
        for offset in range(0, len(epoch_data[0]), BATCH_SIZE):
            batch = slice(offset, offset + BATCH_SIZE)
            visible = (
                torch.from_numpy(epoch_data[0][batch])
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            infrared = (
                torch.from_numpy(epoch_data[1][batch])
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            visible_boxes = torch.from_numpy(epoch_data[2][batch]).to(device)
            infrared_boxes = torch.from_numpy(epoch_data[3][batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            visible_to_infrared = model(infrared, visible, direction="visible_to_infrared")
            infrared_to_visible = model(infrared, visible, direction="infrared_to_visible")
            loss, values = bidirectional_geometry_loss(
                visible,
                infrared,
                visible_boxes,
                infrared_boxes,
                visible_to_infrared,
                infrared_to_visible,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite v3 loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
            "loss": {key: value / steps_per_epoch for key, value in sorted(totals.items())},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if (epoch + 1) % 10 == 0 or epoch + 1 == EPOCHS:
            _save_checkpoint(resume_path, model, optimizer, scheduler, epoch + 1, metadata)
    final_path = args.output_dir / "antiuav300_dense_matcher_bidirectional_v3_e300.pth"
    _save_checkpoint(final_path, model, optimizer, scheduler, EPOCHS, metadata)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--pairs-per-sequence", type=int, default=PAIRS_PER_SEQUENCE_PER_EPOCH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    frozen = (args.epochs, args.pairs_per_sequence, args.batch_size, args.learning_rate)
    expected = (EPOCHS, PAIRS_PER_SEQUENCE_PER_EPOCH, BATCH_SIZE, LEARNING_RATE)
    if frozen != expected:
        parser.error(f"protocol v3 is frozen at epochs/pairs/batch/lr={expected}")
    project_root = Path(__file__).resolve().parents[1]
    args.root = args.root.resolve()
    args.initial_checkpoint = args.initial_checkpoint.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.root.is_dir() or not args.initial_checkpoint.is_file():
        parser.error("dataset root and v2 checkpoint must exist")
    if file_sha256(args.initial_checkpoint) != INITIAL_CHECKPOINT_SHA256:
        parser.error("protocol v3 requires the frozen v2 300-epoch checkpoint")
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
    print("Preparing/verifying complete train-only v3 cache...", flush=True)
    cache_manifest = _build_cache(args.root, args.cache_root)
    print(json.dumps(cache_manifest, indent=2, sort_keys=True), flush=True)
    final_path = _train(args, cache_manifest, project_root)
    result = {
        "checkpoint": str(final_path),
        "checkpoint_sha256": file_sha256(final_path),
        "training_metadata": torch.load(final_path, map_location="cpu", weights_only=True)[
            "aero_registration"
        ],
    }
    result["training_result_sha256"] = canonical_hash(result)
    result_path = args.output_dir / "training_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
