#!/usr/bin/env python3
"""Audit image-conditioned Anti-UAV300 visible-to-IR dense registration."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from aero_ir.data.antiuav import ANTIUAV300_SPLITS, load_split_manifest
from aero_ir.registration.superfusion import (
    load_superfusion_matcher,
    transform_boxes_source_to_target,
    warp_source_to_target,
)
from aero_ir.utils.manifest import canonical_hash, file_sha256

SCHEMA_VERSION = 1
CHECKPOINT_SHA256 = "09337dddeb5c80b5c6b484f93298c355dd2a499b47260a9a0e03ae11fdf9a416"
MIN_IOU = 0.6
MAX_CENTROID_SHIFT_FRACTION = 0.25
MAX_AREA_RATIO_CHANGE = 0.5
MIN_FRAME_PASS_RATE = 0.95
MIN_VALID_FLOW_FRACTION = 0.90
MAX_INVERSE_RESIDUAL = 0.01
MIN_EDGE_NCC_MEDIAN_GAIN = 0.02
MIN_EDGE_NCC_IMPROVEMENT_RATE = 0.75
NETWORK_SIZE = 256


@dataclass(frozen=True)
class FramePair:
    split: str
    sequence_id: str
    frame_index: int
    visible: np.ndarray
    infrared: np.ndarray
    source_box: np.ndarray
    target_box: np.ndarray


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _valid_box(box: object, width: int, height: int) -> np.ndarray | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return None
    if (
        box_width <= 0
        or box_height <= 0
        or x < 0
        or y < 0
        or x + box_width > width + 1e-6
        or y + box_height > height + 1e-6
    ):
        return None
    return np.asarray(
        [
            (x + box_width / 2.0) / width,
            (y + box_height / 2.0) / height,
            box_width / width,
            box_height / height,
        ],
        dtype=np.float32,
    )


def _candidate_pairs(sequence_root: Path) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    visible = _load_json(sequence_root / "visible.json")
    infrared = _load_json(sequence_root / "infrared.json")
    arrays = (
        visible.get("exist"),
        visible.get("gt_rect"),
        infrared.get("exist"),
        infrared.get("gt_rect"),
    )
    if not all(isinstance(values, list) for values in arrays):
        raise ValueError(f"invalid paired annotations in {sequence_root}")
    if len({len(values) for values in arrays}) != 1:
        raise ValueError(f"paired annotation length mismatch in {sequence_root}")
    indices: list[int] = []
    source_boxes: list[np.ndarray] = []
    target_boxes: list[np.ndarray] = []
    for index, (visible_exists, visible_box, infrared_exists, infrared_box) in enumerate(
        zip(*arrays, strict=True)
    ):
        if visible_exists != 1 or infrared_exists != 1:
            continue
        source = _valid_box(visible_box, 1920, 1080)
        target = _valid_box(infrared_box, 640, 512)
        if source is None or target is None:
            continue
        indices.append(index)
        source_boxes.append(source)
        target_boxes.append(target)
    return indices, source_boxes, target_boxes


def _balanced_indices(length: int, samples_per_sequence: int) -> np.ndarray:
    if samples_per_sequence == 0 or length <= samples_per_sequence:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, length - 1, samples_per_sequence)).astype(np.int64))


def _read_at(capture: cv2.VideoCapture, frame_index: int, path: Path) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame {frame_index} from {path}")
    return frame


def _prepare_pair(
    *,
    split: str,
    sequence_id: str,
    frame_index: int,
    visible: np.ndarray,
    infrared: np.ndarray,
    source_box: np.ndarray,
    target_box: np.ndarray,
    sequence_root: Path,
) -> FramePair:
    if visible.shape[:2] != (1080, 1920):
        raise ValueError(f"unexpected visible frame shape in {sequence_root}")
    if infrared.shape[:2] != (512, 640):
        raise ValueError(f"unexpected infrared frame shape in {sequence_root}")
    return FramePair(
        split=split,
        sequence_id=sequence_id,
        frame_index=frame_index,
        visible=cv2.cvtColor(
            cv2.resize(visible, (NETWORK_SIZE, NETWORK_SIZE)),
            cv2.COLOR_BGR2RGB,
        ),
        infrared=cv2.cvtColor(
            cv2.resize(infrared, (NETWORK_SIZE, NETWORK_SIZE)),
            cv2.COLOR_BGR2RGB,
        ),
        source_box=source_box,
        target_box=target_box,
    )


def _iter_pairs(
    root: Path,
    splits: tuple[str, ...],
    samples_per_sequence: int,
) -> Iterator[FramePair]:
    for split in splits:
        for sequence_id in sorted(load_split_manifest(root, split)):
            sequence_root = root / split / sequence_id
            indices, source_boxes, target_boxes = _candidate_pairs(sequence_root)
            if not indices:
                continue
            selected = _balanced_indices(len(indices), samples_per_sequence)
            visible_path = sequence_root / "visible.mp4"
            infrared_path = sequence_root / "infrared.mp4"
            visible_capture = cv2.VideoCapture(str(visible_path))
            infrared_capture = cv2.VideoCapture(str(infrared_path))
            if not visible_capture.isOpened() or not infrared_capture.isOpened():
                raise RuntimeError(f"could not open paired videos in {sequence_root}")
            try:
                if samples_per_sequence == 0:
                    selected_positions = iter(selected.tolist())
                    next_position = next(selected_positions, None)
                    for frame_index in range(indices[-1] + 1):
                        visible_ok, visible = visible_capture.read()
                        infrared_ok, infrared = infrared_capture.read()
                        if not visible_ok or visible is None:
                            raise RuntimeError(
                                f"could not decode frame {frame_index} from {visible_path}"
                            )
                        if not infrared_ok or infrared is None:
                            raise RuntimeError(
                                f"could not decode frame {frame_index} from {infrared_path}"
                            )
                        if next_position is not None and frame_index == indices[next_position]:
                            yield _prepare_pair(
                                split=split,
                                sequence_id=sequence_id,
                                frame_index=frame_index,
                                visible=visible,
                                infrared=infrared,
                                source_box=source_boxes[next_position],
                                target_box=target_boxes[next_position],
                                sequence_root=sequence_root,
                            )
                            next_position = next(selected_positions, None)
                else:
                    for raw_position in selected:
                        position = int(raw_position)
                        frame_index = indices[position]
                        yield _prepare_pair(
                            split=split,
                            sequence_id=sequence_id,
                            frame_index=frame_index,
                            visible=_read_at(visible_capture, frame_index, visible_path),
                            infrared=_read_at(infrared_capture, frame_index, infrared_path),
                            source_box=source_boxes[position],
                            target_box=target_boxes[position],
                            sequence_root=sequence_root,
                        )
            finally:
                visible_capture.release()
                infrared_capture.release()


def _batches(pairs: Iterator[FramePair], batch_size: int) -> Iterator[list[FramePair]]:
    batch: list[FramePair] = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _edge_map(images: torch.Tensor) -> torch.Tensor:
    grayscale = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).reshape(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gradient_x = F.conv2d(grayscale, kernel_x, padding=1)
    gradient_y = F.conv2d(grayscale, kernel_y, padding=1)
    return torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)


def _edge_ncc(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_edges = _edge_map(source)[:, :, 38:243, 13:243].flatten(1)
    target_edges = _edge_map(target)[:, :, 38:243, 13:243].flatten(1)
    source_edges = source_edges - source_edges.mean(dim=1, keepdim=True)
    target_edges = target_edges - target_edges.mean(dim=1, keepdim=True)
    numerator = (source_edges * target_edges).sum(dim=1)
    denominator = torch.linalg.vector_norm(source_edges, dim=1) * torch.linalg.vector_norm(
        target_edges, dim=1
    )
    return numerator / denominator.clamp_min(1e-12)


def _flow_valid_fraction(displacement: torch.Tensor) -> torch.Tensor:
    horizontal = torch.linspace(
        -1.0,
        1.0,
        displacement.shape[3],
        device=displacement.device,
        dtype=displacement.dtype,
    )
    vertical = torch.linspace(
        -1.0,
        1.0,
        displacement.shape[2],
        device=displacement.device,
        dtype=displacement.dtype,
    )
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)
    flow = grid + displacement
    return ((flow >= -1.0) & (flow <= 1.0)).all(dim=1).float().mean(dim=(1, 2))


def _bbox_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    predicted_min = predicted[:, :2] - predicted[:, 2:] / 2.0
    predicted_max = predicted[:, :2] + predicted[:, 2:] / 2.0
    target_min = target[:, :2] - target[:, 2:] / 2.0
    target_max = target[:, :2] + target[:, 2:] / 2.0
    intersection_size = np.maximum(
        0.0,
        np.minimum(predicted_max, target_max) - np.maximum(predicted_min, target_min),
    )
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    predicted_area = predicted[:, 2] * predicted[:, 3]
    target_area = target[:, 2] * target[:, 3]
    union = predicted_area + target_area - intersection
    iou = intersection / np.maximum(union, 1e-12)
    center_error = np.linalg.norm(predicted[:, :2] - target[:, :2], axis=1)
    target_diagonal = np.linalg.norm(target[:, 2:], axis=1)
    shift_fraction = center_error / np.maximum(target_diagonal, 1e-12)
    area_change = np.abs(predicted_area / np.maximum(target_area, 1e-12) - 1.0)
    in_bounds = np.all(predicted_min >= 0.0, axis=1) & np.all(predicted_max <= 1.0, axis=1)
    joint = (
        (iou >= MIN_IOU)
        & (shift_fraction <= MAX_CENTROID_SHIFT_FRACTION)
        & (area_change <= MAX_AREA_RATIO_CHANGE)
        & in_bounds
    )
    return {
        "bbox_iou": iou,
        "centroid_shift_fraction": shift_fraction,
        "absolute_area_ratio_change": area_change,
        "in_bounds": in_bounds,
        "joint": joint,
    }


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _split_report(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("dense registration produced no rows for a requested split")
    predicted = np.stack([row["predicted_box"] for row in rows])
    target = np.stack([row["target_box"] for row in rows])
    box_metrics = _bbox_metrics(predicted, target)
    inverse_residual = np.asarray([row["inverse_residual"] for row in rows])
    valid_flow = np.asarray([row["valid_flow_fraction"] for row in rows])
    edge_before = np.asarray([row["edge_ncc_before"] for row in rows])
    edge_after = np.asarray([row["edge_ncc_after"] for row in rows])
    edge_gain = edge_after - edge_before
    sequence_rates = []
    per_sequence = {}
    for sequence_id in sorted({row["sequence_id"] for row in rows}):
        mask = np.asarray([row["sequence_id"] == sequence_id for row in rows])
        joint_rate = float(np.mean(box_metrics["joint"][mask]))
        sequence_rates.append(joint_rate)
        per_sequence[sequence_id] = {
            "pairs": int(mask.sum()),
            "joint_pass_rate": joint_rate,
            "median_bbox_iou": float(np.median(box_metrics["bbox_iou"][mask])),
            "median_edge_ncc_gain": float(np.median(edge_gain[mask])),
        }
    worst_indices = np.argsort(box_metrics["bbox_iou"], kind="stable")[:20]
    return {
        "counts": {
            "sequences": len({row["sequence_id"] for row in rows}),
            "evaluated_pairs": len(rows),
        },
        "bbox_iou": _summary(box_metrics["bbox_iou"]),
        "centroid_shift_fraction": _summary(box_metrics["centroid_shift_fraction"]),
        "absolute_area_ratio_change": _summary(box_metrics["absolute_area_ratio_change"]),
        "inverse_solver_residual": _summary(inverse_residual),
        "valid_flow_fraction": _summary(valid_flow),
        "edge_ncc_before": _summary(edge_before),
        "edge_ncc_after": _summary(edge_after),
        "edge_ncc_gain": _summary(edge_gain),
        "frame_pass_rate": {
            "iou": float(np.mean(box_metrics["bbox_iou"] >= MIN_IOU)),
            "centroid_shift": float(
                np.mean(box_metrics["centroid_shift_fraction"] <= MAX_CENTROID_SHIFT_FRACTION)
            ),
            "area": float(
                np.mean(box_metrics["absolute_area_ratio_change"] <= MAX_AREA_RATIO_CHANGE)
            ),
            "in_bounds": float(np.mean(box_metrics["in_bounds"])),
            "joint": float(np.mean(box_metrics["joint"])),
            "valid_flow": float(np.mean(valid_flow >= MIN_VALID_FLOW_FRACTION)),
            "inverse_residual": float(np.mean(inverse_residual <= MAX_INVERSE_RESIDUAL)),
            "edge_ncc_improved": float(np.mean(edge_gain > 0.0)),
        },
        "sequence_macro_joint_pass_rate": _summary(np.asarray(sequence_rates)),
        "per_sequence": per_sequence,
        "worst_bbox_iou_examples": [
            {
                "sequence_id": rows[int(index)]["sequence_id"],
                "frame_index": rows[int(index)]["frame_index"],
                "bbox_iou": float(box_metrics["bbox_iou"][index]),
                "centroid_shift_fraction": float(box_metrics["centroid_shift_fraction"][index]),
                "absolute_area_ratio_change": float(
                    box_metrics["absolute_area_ratio_change"][index]
                ),
            }
            for index in worst_indices
        ],
    }


def _evaluate(
    root: Path,
    checkpoint: Path,
    splits: tuple[str, ...],
    samples_per_sequence: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, list[dict]], float]:
    model = load_superfusion_matcher(checkpoint, device)
    rows: dict[str, list[dict]] = defaultdict(list)
    started = time.perf_counter()
    processed = 0
    pairs = _iter_pairs(root, splits, samples_per_sequence)
    with torch.inference_mode():
        for batch in _batches(pairs, batch_size):
            source = (
                torch.from_numpy(np.stack([pair.visible for pair in batch]))
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            target = (
                torch.from_numpy(np.stack([pair.infrared for pair in batch]))
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            source_boxes = torch.from_numpy(np.stack([pair.source_box for pair in batch])).to(
                device=device, dtype=torch.float32
            )
            displacement = model(target, source, direction="visible_to_infrared")
            predicted_boxes, inverse_residual = transform_boxes_source_to_target(
                source_boxes, displacement
            )
            registered = warp_source_to_target(source, displacement)
            edge_before = _edge_ncc(source, target)
            edge_after = _edge_ncc(registered, target)
            valid_flow = _flow_valid_fraction(displacement)
            predicted_numpy = predicted_boxes.cpu().numpy()
            residual_numpy = inverse_residual.cpu().numpy()
            valid_numpy = valid_flow.cpu().numpy()
            edge_before_numpy = edge_before.cpu().numpy()
            edge_after_numpy = edge_after.cpu().numpy()
            for index, pair in enumerate(batch):
                rows[pair.split].append(
                    {
                        "sequence_id": pair.sequence_id,
                        "frame_index": pair.frame_index,
                        "source_box": pair.source_box.tolist(),
                        "target_box": pair.target_box.tolist(),
                        "predicted_box": predicted_numpy[index].tolist(),
                        "inverse_residual": float(residual_numpy[index]),
                        "valid_flow_fraction": float(valid_numpy[index]),
                        "edge_ncc_before": float(edge_before_numpy[index]),
                        "edge_ncc_after": float(edge_after_numpy[index]),
                    }
                )
            processed += len(batch)
            if processed % max(batch_size * 20, batch_size) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"processed {processed} pairs ({processed / max(elapsed, 1e-9):.1f}/s)",
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return dict(rows), time.perf_counter() - started


def _gate_report(metrics: dict[str, dict], exhaustive: bool) -> dict[str, str]:
    train = metrics.get("train")
    validation = metrics.get("val")
    train_pass = bool(train and train["frame_pass_rate"]["joint"] >= MIN_FRAME_PASS_RATE)
    validation_pass = bool(
        validation and validation["frame_pass_rate"]["joint"] >= MIN_FRAME_PASS_RATE
    )
    dense_validation_pass = bool(
        validation_pass
        and validation["frame_pass_rate"]["valid_flow"] >= MIN_FRAME_PASS_RATE
        and validation["frame_pass_rate"]["inverse_residual"] >= MIN_FRAME_PASS_RATE
        and validation["edge_ncc_gain"]["p50"] >= MIN_EDGE_NCC_MEDIAN_GAIN
        and validation["frame_pass_rate"]["edge_ncc_improved"] >= MIN_EDGE_NCC_IMPROVEMENT_RATE
    )
    screening_pass = train_pass and dense_validation_pass
    if not exhaustive:
        return {
            "sequence_balanced_screen": "pass" if screening_pass else "hold",
            "held_out_validation": "candidate" if screening_pass else "hold",
            "dense_paired_image_registration": "hold_pending_exhaustive_audit",
            "generator_training_eligible": "hold_pending_exhaustive_audit",
        }
    return {
        "sequence_balanced_screen": "not_applicable",
        "train_fit_sanity": "pass" if train_pass else "hold",
        "held_out_validation": "pass" if validation_pass else "hold",
        "dense_paired_image_registration": "pass" if screening_pass else "hold",
        "generator_training_eligible": "pass" if screening_pass else "hold",
    }


def _checkpoint_provenance(checkpoint: Path, checkpoint_sha256: str) -> dict:
    if checkpoint_sha256 == CHECKPOINT_SHA256:
        return {
            "checkpoint_variant": "public_RoadScene",
            "checkpoint_sha256": checkpoint_sha256,
            "anti_uav_fitting": "none",
        }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("aero_registration") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("unrecognized dense-registration checkpoint")
    required = {
        "kind": "antiuav300_train_only_superfusion_finetune",
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "initial_checkpoint_sha256": CHECKPOINT_SHA256,
        "epochs": 300,
        "precision": "float32",
        "seed": 0,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("fine-tuned registration checkpoint violates the frozen protocol")
    if not isinstance(metadata.get("cache_manifest_sha256"), str):
        raise ValueError("fine-tuned registration checkpoint has no cache provenance")
    return {
        "checkpoint_variant": "Anti-UAV300_train_only_finetune",
        "checkpoint_sha256": checkpoint_sha256,
        "anti_uav_fitting": "official train split only",
        "training_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ANTIUAV300_SPLITS,
        default=["train", "val"],
    )
    parser.add_argument(
        "--samples-per-sequence",
        type=int,
        default=8,
        help="0 evaluates every usable paired target frame",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.samples_per_sequence < 0:
        parser.error("--samples-per-sequence must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    splits = tuple(dict.fromkeys(args.splits))
    if "val" not in splits:
        parser.error("held-out validation is required for registration qualification")
    if args.samples_per_sequence == 0 and "train" not in splits:
        parser.error("an exhaustive qualification requires train and val")

    root = args.root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Anti-UAV300 root is missing: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SuperFusion checkpoint is missing: {checkpoint}")
    checkpoint_sha256 = file_sha256(checkpoint)
    checkpoint_provenance = _checkpoint_provenance(checkpoint, checkpoint_sha256)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.use_deterministic_algorithms(True)

    rows, elapsed_seconds = _evaluate(
        root,
        checkpoint,
        splits,
        args.samples_per_sequence,
        args.batch_size,
        device,
    )
    metrics = {split: _split_report(rows[split]) for split in splits}
    exhaustive = args.samples_per_sequence == 0
    import kornia

    provenance = {
        split: {
            "split_manifest_sha256": file_sha256(root / "label_new" / f"{split}.json"),
            "evaluation_records_sha256": canonical_hash(rows[split]),
        }
        for split in splits
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_superfusion_dense_registration_audit",
        "dataset": "Anti-UAV300",
        "root": str(root),
        "model": {
            "architecture": "SuperFusion DenseMatcher",
            "upstream_revision": "bee015a8938bee549d0132b80c1daf161ff80660",
            "direction": "visible_to_infrared",
            "network_input_size": [NETWORK_SIZE, NETWORK_SIZE],
            "precision": "float32",
            **checkpoint_provenance,
        },
        "data_usage": {
            "model_fit": (
                "official train only"
                if checkpoint_provenance["anti_uav_fitting"] != "none"
                else "public RoadScene checkpoint; no Anti-UAV300 frames or labels"
            ),
            "qualification": "official validation; never used to fit or select parameters",
            "test": "report only when requested; never used to fit, select, or qualify",
            "frame_selection": (
                "all usable paired target frames"
                if exhaustive
                else f"{args.samples_per_sequence} endpoint-inclusive usable pairs per sequence"
            ),
        },
        "thresholds": {
            "min_bbox_iou": MIN_IOU,
            "max_centroid_shift_fraction_of_target_diagonal": (MAX_CENTROID_SHIFT_FRACTION),
            "max_absolute_area_ratio_change": MAX_AREA_RATIO_CHANGE,
            "min_joint_frame_pass_rate": MIN_FRAME_PASS_RATE,
            "min_valid_flow_fraction": MIN_VALID_FLOW_FRACTION,
            "max_inverse_solver_residual": MAX_INVERSE_RESIDUAL,
            "min_validation_median_edge_ncc_gain": MIN_EDGE_NCC_MEDIAN_GAIN,
            "min_validation_edge_ncc_improvement_rate": MIN_EDGE_NCC_IMPROVEMENT_RATE,
        },
        "source_provenance": provenance,
        "metrics": metrics,
        "gates": _gate_report(metrics, exhaustive),
        "runtime": {
            "device": str(device),
            "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
            "torch_version": torch.__version__,
            "kornia_version": kornia.__version__,
            "batch_size": args.batch_size,
            "elapsed_seconds": elapsed_seconds,
            "pairs_per_second": sum(len(value) for value in rows.values()) / elapsed_seconds,
        },
        "notes": [
            "All gate thresholds are fixed in source before this audit runs.",
            "Visible and IR frames are resized independently to the checkpoint's square input.",
            "Sub-pixel boxes are transformed by inversion of the target-to-source sampling field.",
            "Top telemetry overlays and image borders are excluded from edge-NCC measurements.",
            "A sequence-balanced screen cannot clear the exhaustive registration gate.",
        ],
    }
    report["dense_registration_audit_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
