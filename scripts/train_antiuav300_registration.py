#!/usr/bin/env python3
"""Fine-tune SuperFusion registration on Anti-UAV300 train sequences only."""

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
from torch.nn import functional as F

from aero_ir.data.antiuav import load_split_manifest
from aero_ir.registration.superfusion import (
    box_perimeter_points,
    load_superfusion_matcher,
    warp_source_to_target,
)
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import (
    CHECKPOINT_SHA256,
    NETWORK_SIZE,
    _balanced_indices,
    _candidate_pairs,
    _edge_ncc,
    _prepare_pair,
    _read_at,
)

SCHEMA_VERSION = 1
DEFAULT_EPOCHS = 300
DEFAULT_SAMPLES_PER_SEQUENCE = 16
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
POINT_WEIGHT = 10.0
EDGE_WEIGHT = 0.25
SMOOTHNESS_WEIGHT = 0.05
MAGNITUDE_WEIGHT = 0.01
BOUNDARY_WEIGHT = 0.10


def _git_sha(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _cache_paths(cache_root: Path) -> dict[str, Path]:
    return {
        "visible": cache_root / "visible.npy",
        "infrared": cache_root / "infrared.npy",
        "source_boxes": cache_root / "source_boxes.npy",
        "target_boxes": cache_root / "target_boxes.npy",
        "manifest": cache_root / "manifest.json",
    }


def _build_cache(root: Path, cache_root: Path, samples_per_sequence: int) -> dict:
    paths = _cache_paths(cache_root)
    if paths["manifest"].is_file():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("samples_per_sequence") != samples_per_sequence:
            raise ValueError("existing registration cache uses a different sample count")
        for key in ("visible", "infrared", "source_boxes", "target_boxes"):
            if not paths[key].is_file() or file_sha256(paths[key]) != manifest["artifacts"][key]:
                raise ValueError(f"registration cache artifact is missing or changed: {paths[key]}")
        return manifest

    sequences = sorted(load_split_manifest(root, "train"))
    selections = []
    annotation_inputs = []
    for sequence_id in sequences:
        sequence_root = root / "train" / sequence_id
        indices, source_boxes, target_boxes = _candidate_pairs(sequence_root)
        selected = _balanced_indices(len(indices), samples_per_sequence)
        selections.append((sequence_id, indices, source_boxes, target_boxes, selected))
        annotation_inputs.append(
            {
                "sequence_id": sequence_id,
                "visible_sha256": file_sha256(sequence_root / "visible.json"),
                "infrared_sha256": file_sha256(sequence_root / "infrared.json"),
            }
        )
    count = sum(len(selection[-1]) for selection in selections)
    cache_root.mkdir(parents=True, exist_ok=False)
    visible_cache = np.lib.format.open_memmap(
        paths["visible"], mode="w+", dtype=np.uint8, shape=(count, NETWORK_SIZE, NETWORK_SIZE, 3)
    )
    infrared_cache = np.lib.format.open_memmap(
        paths["infrared"],
        mode="w+",
        dtype=np.uint8,
        shape=(count, NETWORK_SIZE, NETWORK_SIZE, 3),
    )
    source_box_cache = np.lib.format.open_memmap(
        paths["source_boxes"], mode="w+", dtype=np.float32, shape=(count, 4)
    )
    target_box_cache = np.lib.format.open_memmap(
        paths["target_boxes"], mode="w+", dtype=np.float32, shape=(count, 4)
    )
    records = []
    cache_index = 0
    for sequence_number, selection in enumerate(selections, start=1):
        sequence_id, indices, source_boxes, target_boxes, selected = selection
        sequence_root = root / "train" / sequence_id
        visible_path = sequence_root / "visible.mp4"
        infrared_path = sequence_root / "infrared.mp4"
        visible_capture = cv2.VideoCapture(str(visible_path))
        infrared_capture = cv2.VideoCapture(str(infrared_path))
        if not visible_capture.isOpened() or not infrared_capture.isOpened():
            raise RuntimeError(f"could not open paired videos in {sequence_root}")
        try:
            for raw_position in selected:
                position = int(raw_position)
                frame_index = indices[position]
                pair = _prepare_pair(
                    split="train",
                    sequence_id=sequence_id,
                    frame_index=frame_index,
                    visible=_read_at(visible_capture, frame_index, visible_path),
                    infrared=_read_at(infrared_capture, frame_index, infrared_path),
                    source_box=source_boxes[position],
                    target_box=target_boxes[position],
                    sequence_root=sequence_root,
                )
                visible_cache[cache_index] = pair.visible
                infrared_cache[cache_index] = pair.infrared
                source_box_cache[cache_index] = pair.source_box
                target_box_cache[cache_index] = pair.target_box
                records.append(
                    {
                        "cache_index": cache_index,
                        "sequence_id": sequence_id,
                        "frame_index": frame_index,
                    }
                )
                cache_index += 1
        finally:
            visible_capture.release()
            infrared_capture.release()
        print(f"cached train sequence {sequence_number}/{len(sequences)}", flush=True)
    for array in (visible_cache, infrared_cache, source_box_cache, target_box_cache):
        array.flush()
    del visible_cache, infrared_cache, source_box_cache, target_box_cache
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_dense_registration_train_cache",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "network_size": [NETWORK_SIZE, NETWORK_SIZE],
        "samples_per_sequence": samples_per_sequence,
        "sequences": len(sequences),
        "pairs": count,
        "selection": "endpoint-inclusive usable target pairs per official training sequence",
        "annotation_set_sha256": canonical_hash(annotation_inputs),
        "record_set_sha256": canonical_hash(records),
        "artifacts": {
            key: file_sha256(paths[key])
            for key in ("visible", "infrared", "source_boxes", "target_boxes")
        },
    }
    manifest["cache_manifest_sha256"] = canonical_hash(manifest)
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _sampling_grid(reference: torch.Tensor) -> torch.Tensor:
    height, width = reference.shape[2:]
    vertical = torch.linspace(-1.0, 1.0, height, device=reference.device)
    horizontal = torch.linspace(-1.0, 1.0, width, device=reference.device)
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)


def _registration_loss(
    visible: torch.Tensor,
    infrared: torch.Tensor,
    source_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    displacement: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    source_points = box_perimeter_points(source_boxes)
    target_points = box_perimeter_points(target_boxes)
    sampled_displacement = (
        F.grid_sample(
            displacement,
            target_points.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        .squeeze(-1)
        .permute(0, 2, 1)
    )
    predicted_source_points = target_points + sampled_displacement
    point_loss = F.smooth_l1_loss(
        predicted_source_points,
        source_points,
        beta=0.01,
    )
    registered = warp_source_to_target(visible, displacement)
    edge_loss = 1.0 - _edge_ncc(registered, infrared).mean()
    horizontal_smoothness = (displacement[:, :, :, 1:] - displacement[:, :, :, :-1]).abs().mean()
    vertical_smoothness = (displacement[:, :, 1:, :] - displacement[:, :, :-1, :]).abs().mean()
    smoothness_loss = horizontal_smoothness + vertical_smoothness
    magnitude_loss = displacement.square().mean()
    flow = _sampling_grid(displacement) + displacement
    boundary_loss = F.relu(flow.abs() - 1.0).mean()
    total = (
        POINT_WEIGHT * point_loss
        + EDGE_WEIGHT * edge_loss
        + SMOOTHNESS_WEIGHT * smoothness_loss
        + MAGNITUDE_WEIGHT * magnitude_loss
        + BOUNDARY_WEIGHT * boundary_loss
    )
    values = {
        "total": float(total.detach()),
        "point": float(point_loss.detach()),
        "edge": float(edge_loss.detach()),
        "smoothness": float(smoothness_loss.detach()),
        "magnitude": float(magnitude_loss.detach()),
        "boundary": float(boundary_loss.detach()),
    }
    return total, values


def _save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metadata: dict,
) -> None:
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
    paths = _cache_paths(args.cache_root)
    visible_cache = np.load(paths["visible"], mmap_mode="r")
    infrared_cache = np.load(paths["infrared"], mmap_mode="r")
    source_box_cache = np.load(paths["source_boxes"], mmap_mode="r")
    target_box_cache = np.load(paths["target_boxes"], mmap_mode="r")
    count = len(visible_cache)
    if not all(
        len(array) == count for array in (infrared_cache, source_box_cache, target_box_cache)
    ):
        raise ValueError("registration cache arrays have inconsistent lengths")
    if count % args.batch_size:
        raise ValueError("cached pair count must be divisible by batch size")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = load_superfusion_matcher(args.initial_checkpoint, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    steps_per_epoch = count // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * steps_per_epoch,
        eta_min=1e-6,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_train_only_superfusion_finetune",
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "initial_checkpoint_sha256": file_sha256(args.initial_checkpoint),
        "cache_manifest_sha256": cache_manifest["cache_manifest_sha256"],
        "git_sha": _git_sha(project_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": args.epochs * steps_per_epoch,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": 1e-6,
        "optimizer": "AdamW",
        "loss_weights": {
            "paired_box_perimeter": POINT_WEIGHT,
            "edge_ncc": EDGE_WEIGHT,
            "flow_smoothness": SMOOTHNESS_WEIGHT,
            "flow_magnitude": MAGNITUDE_WEIGHT,
            "flow_boundary": BOUNDARY_WEIGHT,
        },
        "precision": "float32",
        "seed": args.seed,
        "determinism": (
            "fixed seeds, deterministic cuDNN/TF32 disabled; deterministic-algorithm warnings "
            "enabled because CUDA grid_sample backward has no deterministic implementation"
        ),
    }
    start_epoch = 0
    resume_path = args.output_dir / "latest.pth"
    if args.resume and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload.get("aero_registration") != metadata:
            raise ValueError("resume checkpoint metadata does not match this training run")
        model.load_state_dict(payload["DM"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"])
        print(f"resuming after epoch {start_epoch}", flush=True)
    elif args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing training directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    generator = torch.Generator(device="cpu")
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        generator.manual_seed(args.seed + epoch)
        permutation = torch.randperm(count, generator=generator).numpy()
        totals: dict[str, float] = {}
        for offset in range(0, count, args.batch_size):
            indices = permutation[offset : offset + args.batch_size]
            visible = (
                torch.from_numpy(np.asarray(visible_cache[indices]).copy())
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            infrared = (
                torch.from_numpy(np.asarray(infrared_cache[indices]).copy())
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            source_boxes = torch.from_numpy(np.asarray(source_box_cache[indices]).copy()).to(
                device=device
            )
            target_boxes = torch.from_numpy(np.asarray(target_box_cache[indices]).copy()).to(
                device=device
            )
            optimizer.zero_grad(set_to_none=True)
            displacement = model(infrared, visible, direction="visible_to_infrared")
            loss, values = _registration_loss(
                visible,
                infrared,
                source_boxes,
                target_boxes,
                displacement,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite registration loss at epoch {epoch + 1}")
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
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            _save_training_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metadata=metadata,
            )

    final_path = args.output_dir / f"antiuav300_dense_matcher_e{args.epochs}.pth"
    _save_training_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        metadata=metadata,
    )
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--samples-per-sequence", type=int, default=DEFAULT_SAMPLES_PER_SEQUENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs != DEFAULT_EPOCHS:
        parser.error(f"the frozen registration protocol requires {DEFAULT_EPOCHS} epochs")
    if args.samples_per_sequence != DEFAULT_SAMPLES_PER_SEQUENCE:
        parser.error(
            "the frozen registration protocol requires "
            f"{DEFAULT_SAMPLES_PER_SEQUENCE} pairs per training sequence"
        )
    if args.batch_size != DEFAULT_BATCH_SIZE:
        parser.error(f"the frozen registration protocol requires batch={DEFAULT_BATCH_SIZE}")
    if args.learning_rate != DEFAULT_LEARNING_RATE:
        parser.error(f"the frozen registration protocol requires lr={DEFAULT_LEARNING_RATE}")

    project_root = Path(__file__).resolve().parents[1]
    args.root = args.root.resolve()
    args.initial_checkpoint = args.initial_checkpoint.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.root.is_dir():
        raise FileNotFoundError(f"Anti-UAV300 root is missing: {args.root}")
    if not args.initial_checkpoint.is_file():
        raise FileNotFoundError(f"initial checkpoint is missing: {args.initial_checkpoint}")
    initial_sha256 = file_sha256(args.initial_checkpoint)
    if initial_sha256 != CHECKPOINT_SHA256:
        raise ValueError(
            f"initial checkpoint hash mismatch: {initial_sha256} != {CHECKPOINT_SHA256}"
        )

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

    print("Preparing/verifying train-only registration cache...", flush=True)
    cache_manifest = _build_cache(args.root, args.cache_root, args.samples_per_sequence)
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
