#!/usr/bin/env python3
"""Train frozen Anti-UAV300 registration protocol v2 on train sequences only."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.data.antiuav import load_split_manifest
from aero_ir.registration.protocol_v2 import (
    BATCH_SIZE,
    EPOCHS,
    INITIAL_CHECKPOINT_SHA256,
    LEARNING_RATE,
    LOSS_WEIGHTS,
    PAIRS_PER_SEQUENCE_PER_EPOCH,
    geometry_first_loss,
    rotating_positions,
)
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import (
    NETWORK_SIZE,
    _candidate_pairs,
    _prepare_pair,
)

SCHEMA_VERSION = 2


def _git_sha(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _shard_paths(cache_root: Path, sequence_id: str) -> dict[str, Path]:
    shard_root = cache_root / "shards" / sequence_id
    return {
        "root": shard_root,
        "visible": shard_root / "visible.npy",
        "infrared": shard_root / "infrared.npy",
        "source_boxes": shard_root / "source_boxes.npy",
        "target_boxes": shard_root / "target_boxes.npy",
        "manifest": shard_root / "manifest.json",
    }


def _verify_shard(paths: dict[str, Path], sequence_id: str, count: int) -> dict | None:
    if not paths["manifest"].is_file():
        return None
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("sequence_id") != sequence_id or manifest.get("pairs") != count:
        raise ValueError(f"cache shard metadata mismatch: {sequence_id}")
    for key in ("visible", "infrared", "source_boxes", "target_boxes"):
        if not paths[key].is_file() or file_sha256(paths[key]) != manifest["artifacts"][key]:
            raise ValueError(f"cache shard artifact mismatch: {paths[key]}")
    unhashed = {key: value for key, value in manifest.items() if key != "shard_sha256"}
    if canonical_hash(unhashed) != manifest.get("shard_sha256"):
        raise ValueError(f"cache shard manifest hash mismatch: {sequence_id}")
    return manifest


def _build_shard(root: Path, cache_root: Path, sequence_id: str) -> dict:
    sequence_root = root / "train" / sequence_id
    indices, source_boxes, target_boxes = _candidate_pairs(sequence_root)
    count = len(indices)
    if count < PAIRS_PER_SEQUENCE_PER_EPOCH:
        raise ValueError(f"too few usable train pairs in {sequence_id}: {count}")
    paths = _shard_paths(cache_root, sequence_id)
    existing = _verify_shard(paths, sequence_id, count)
    if existing is not None:
        return existing
    if paths["root"].exists() and any(paths["root"].iterdir()):
        raise FileExistsError(f"partial cache shard requires manual inspection: {paths['root']}")
    paths["root"].mkdir(parents=True, exist_ok=True)
    visible_cache = np.lib.format.open_memmap(
        paths["visible"], mode="w+", dtype=np.uint8, shape=(count, NETWORK_SIZE, NETWORK_SIZE, 3)
    )
    infrared_cache = np.lib.format.open_memmap(
        paths["infrared"], mode="w+", dtype=np.uint8, shape=(count, NETWORK_SIZE, NETWORK_SIZE, 3)
    )
    source_cache = np.lib.format.open_memmap(
        paths["source_boxes"], mode="w+", dtype=np.float32, shape=(count, 4)
    )
    target_cache = np.lib.format.open_memmap(
        paths["target_boxes"], mode="w+", dtype=np.float32, shape=(count, 4)
    )
    visible_path = sequence_root / "visible.mp4"
    infrared_path = sequence_root / "infrared.mp4"
    visible_capture = cv2.VideoCapture(str(visible_path))
    infrared_capture = cv2.VideoCapture(str(infrared_path))
    if not visible_capture.isOpened() or not infrared_capture.isOpened():
        raise RuntimeError(f"could not open paired train videos in {sequence_root}")
    position = 0
    try:
        for frame_index in range(indices[-1] + 1):
            visible_ok, visible = visible_capture.read()
            infrared_ok, infrared = infrared_capture.read()
            if not visible_ok or visible is None or not infrared_ok or infrared is None:
                raise RuntimeError(
                    f"could not decode paired frame {frame_index} in {sequence_root}"
                )
            if position >= count or frame_index != indices[position]:
                continue
            pair = _prepare_pair(
                split="train",
                sequence_id=sequence_id,
                frame_index=frame_index,
                visible=visible,
                infrared=infrared,
                source_box=source_boxes[position],
                target_box=target_boxes[position],
                sequence_root=sequence_root,
            )
            visible_cache[position] = pair.visible
            infrared_cache[position] = pair.infrared
            source_cache[position] = pair.source_box
            target_cache[position] = pair.target_box
            position += 1
    finally:
        visible_capture.release()
        infrared_capture.release()
    if position != count:
        raise RuntimeError(f"cached {position}/{count} usable pairs in {sequence_id}")
    for array in (visible_cache, infrared_cache, source_cache, target_cache):
        array.flush()
    del visible_cache, infrared_cache, source_cache, target_cache
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_registration_v2_train_cache_shard",
        "sequence_id": sequence_id,
        "pairs": count,
        "visible_annotations_sha256": file_sha256(sequence_root / "visible.json"),
        "infrared_annotations_sha256": file_sha256(sequence_root / "infrared.json"),
        "artifacts": {
            key: file_sha256(paths[key])
            for key in ("visible", "infrared", "source_boxes", "target_boxes")
        },
    }
    manifest["shard_sha256"] = canonical_hash(manifest)
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _build_cache(root: Path, cache_root: Path) -> dict:
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unhashed = {key: value for key, value in manifest.items() if key != "cache_manifest_sha256"}
        if canonical_hash(unhashed) != manifest.get("cache_manifest_sha256"):
            raise ValueError("v2 cache manifest hash mismatch")
        for shard in manifest["shards"]:
            paths = _shard_paths(cache_root, shard["sequence_id"])
            verified = _verify_shard(paths, shard["sequence_id"], shard["pairs"])
            if verified is None or verified["shard_sha256"] != shard["shard_sha256"]:
                raise ValueError(f"v2 cache shard set changed: {shard['sequence_id']}")
        return manifest
    sequences = sorted(load_split_manifest(root, "train"))
    shards = []
    for index, sequence_id in enumerate(sequences, start=1):
        shard = _build_shard(root, cache_root, sequence_id)
        shards.append(
            {
                "sequence_id": sequence_id,
                "pairs": shard["pairs"],
                "shard_sha256": shard["shard_sha256"],
            }
        )
        print(f"verified v2 train cache shard {index}/{len(sequences)}", flush=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_registration_v2_full_train_cache",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "network_size": [NETWORK_SIZE, NETWORK_SIZE],
        "split_manifest_sha256": file_sha256(root / "label_new" / "train.json"),
        "pairs": sum(shard["pairs"] for shard in shards),
        "sequences": len(shards),
        "shards": shards,
    }
    manifest["cache_manifest_sha256"] = canonical_hash(manifest)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _epoch_arrays(
    cache_root: Path, shards: list[dict], epoch: int, seed: int
) -> tuple[np.ndarray, ...]:
    count = len(shards) * PAIRS_PER_SEQUENCE_PER_EPOCH
    visible = np.empty((count, NETWORK_SIZE, NETWORK_SIZE, 3), dtype=np.uint8)
    infrared = np.empty_like(visible)
    source_boxes = np.empty((count, 4), dtype=np.float32)
    target_boxes = np.empty_like(source_boxes)
    offset = 0
    for shard in shards:
        paths = _shard_paths(cache_root, shard["sequence_id"])
        positions = rotating_positions(shard["pairs"], epoch, shard["sequence_id"], seed=seed)
        destination = slice(offset, offset + PAIRS_PER_SEQUENCE_PER_EPOCH)
        visible[destination] = np.load(paths["visible"], mmap_mode="r")[positions]
        infrared[destination] = np.load(paths["infrared"], mmap_mode="r")[positions]
        source_boxes[destination] = np.load(paths["source_boxes"], mmap_mode="r")[positions]
        target_boxes[destination] = np.load(paths["target_boxes"], mmap_mode="r")[positions]
        offset += PAIRS_PER_SEQUENCE_PER_EPOCH
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = generator.permutation(count)
    return visible[order], infrared[order], source_boxes[order], target_boxes[order]


def _save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "DM": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "aero_registration": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def _train(args: argparse.Namespace, cache_manifest: dict, project_root: Path) -> Path:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = load_superfusion_matcher(args.initial_checkpoint, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    steps_per_epoch = len(cache_manifest["shards"]) * PAIRS_PER_SEQUENCE_PER_EPOCH // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * steps_per_epoch, eta_min=1e-6
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_train_only_geometry_first_v2",
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
        "precision": "float32",
        "seed": args.seed,
        "selection": ("deterministic rotating per-sequence permutations covering every train pair"),
        "determinism": (
            "fixed seeds, deterministic cuDNN, TF32 disabled, grid-sample warnings enabled"
        ),
    }
    resume_path = args.output_dir / "latest.pth"
    start_epoch = 0
    if args.resume and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload.get("aero_registration") != metadata:
            raise ValueError("v2 resume checkpoint metadata does not match")
        model.load_state_dict(payload["DM"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"])
        print(f"resuming v2 after epoch {start_epoch}", flush=True)
    elif args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v2 training directory: {args.output_dir}")
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
            source_boxes = torch.from_numpy(epoch_data[2][batch]).to(device)
            target_boxes = torch.from_numpy(epoch_data[3][batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            displacement = model(infrared, visible, direction="visible_to_infrared")
            loss, values = geometry_first_loss(
                visible, infrared, source_boxes, target_boxes, displacement
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite v2 loss at epoch {epoch + 1}")
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
    final_path = args.output_dir / "antiuav300_dense_matcher_geometry_v2_e300.pth"
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
        parser.error(f"protocol v2 is frozen at epochs/pairs/batch/lr={expected}")
    project_root = Path(__file__).resolve().parents[1]
    args.root = args.root.resolve()
    args.initial_checkpoint = args.initial_checkpoint.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.root.is_dir() or not args.initial_checkpoint.is_file():
        parser.error("dataset root and v1 checkpoint must exist")
    if file_sha256(args.initial_checkpoint) != INITIAL_CHECKPOINT_SHA256:
        parser.error("protocol v2 requires the frozen v1 300-epoch checkpoint")
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
    print("Preparing/verifying complete train-only v2 cache...", flush=True)
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
