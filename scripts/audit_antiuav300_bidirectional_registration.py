#!/usr/bin/env python3
"""Audit frozen Anti-UAV300 bidirectional registration protocol v3."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from aero_ir.registration.protocol_v3 import inverse_consistency_statistics
from aero_ir.registration.superfusion import (
    load_superfusion_matcher,
    transform_boxes_target_to_source,
    warp_source_to_target,
)
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import (
    MAX_AREA_RATIO_CHANGE,
    MAX_CENTROID_SHIFT_FRACTION,
    MIN_EDGE_NCC_IMPROVEMENT_RATE,
    MIN_EDGE_NCC_MEDIAN_GAIN,
    MIN_FRAME_PASS_RATE,
    MIN_IOU,
    MIN_VALID_FLOW_FRACTION,
    _batches,
    _bbox_metrics,
    _edge_ncc,
    _flow_valid_fraction,
    _iter_pairs,
    _summary,
)

SCHEMA_VERSION = 3
MAX_CYCLE_RESIDUAL = 0.01
MIN_POSITIVE_JACOBIAN_FRACTION = 0.99


def _positive_jacobian_fraction(displacement: torch.Tensor) -> torch.Tensor:
    height, width = displacement.shape[2:]
    derivative_x = (displacement[:, :, :, 1:] - displacement[:, :, :, :-1]) * ((width - 1) / 2.0)
    derivative_y = (displacement[:, :, 1:, :] - displacement[:, :, :-1, :]) * ((height - 1) / 2.0)
    determinant = (1.0 + derivative_x[:, 0, :-1]) * (
        1.0 + derivative_y[:, 1, :, :-1]
    ) - derivative_x[:, 1, :-1] * derivative_y[:, 0, :, :-1]
    return (determinant > 0.0).float().mean(dim=(1, 2))


def _direction_report(rows: list[dict], predicted_key: str, target_key: str) -> dict:
    predicted = np.asarray([row[predicted_key] for row in rows])
    target = np.asarray([row[target_key] for row in rows])
    metrics = _bbox_metrics(predicted, target)
    worst = np.argsort(metrics["bbox_iou"], kind="stable")[:20]
    return {
        "bbox_iou": _summary(metrics["bbox_iou"]),
        "centroid_shift_fraction": _summary(metrics["centroid_shift_fraction"]),
        "absolute_area_ratio_change": _summary(metrics["absolute_area_ratio_change"]),
        "frame_pass_rate": {
            "iou": float(np.mean(metrics["bbox_iou"] >= MIN_IOU)),
            "centroid_shift": float(
                np.mean(metrics["centroid_shift_fraction"] <= MAX_CENTROID_SHIFT_FRACTION)
            ),
            "area": float(np.mean(metrics["absolute_area_ratio_change"] <= MAX_AREA_RATIO_CHANGE)),
            "in_bounds": float(np.mean(metrics["in_bounds"])),
            "joint": float(np.mean(metrics["joint"])),
        },
        "worst_bbox_iou_examples": [
            {
                "sequence_id": rows[int(index)]["sequence_id"],
                "frame_index": rows[int(index)]["frame_index"],
                "bbox_iou": float(metrics["bbox_iou"][index]),
            }
            for index in worst
        ],
    }


def _split_report(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("bidirectional registration produced no rows")
    forward_edge_gain = np.asarray([row["forward_edge_gain"] for row in rows])
    reverse_edge_gain = np.asarray([row["reverse_edge_gain"] for row in rows])
    scalar_keys = (
        "forward_valid_flow_fraction",
        "reverse_valid_flow_fraction",
        "forward_positive_jacobian_fraction",
        "reverse_positive_jacobian_fraction",
        "target_cycle_mean",
        "target_cycle_max",
        "source_cycle_mean",
        "source_cycle_max",
        "target_cycle_valid_fraction",
        "source_cycle_valid_fraction",
    )
    report = {
        "counts": {
            "sequences": len({row["sequence_id"] for row in rows}),
            "evaluated_pairs": len(rows),
        },
        "visible_to_infrared_backward_map": _direction_report(
            rows, "predicted_visible_box", "visible_box"
        ),
        "infrared_to_visible_backward_map": _direction_report(
            rows, "predicted_infrared_box", "infrared_box"
        ),
        **{
            key: _summary(np.asarray([row[key] for row in rows], dtype=np.float64))
            for key in scalar_keys
        },
        "forward_edge_ncc_gain": _summary(forward_edge_gain),
        "reverse_edge_ncc_gain": _summary(reverse_edge_gain),
    }
    report["frame_pass_rate"] = {
        "forward_valid_flow": float(
            np.mean(
                np.asarray([row["forward_valid_flow_fraction"] for row in rows])
                >= MIN_VALID_FLOW_FRACTION
            )
        ),
        "reverse_valid_flow": float(
            np.mean(
                np.asarray([row["reverse_valid_flow_fraction"] for row in rows])
                >= MIN_VALID_FLOW_FRACTION
            )
        ),
        "forward_positive_jacobian": float(
            np.mean(
                np.asarray([row["forward_positive_jacobian_fraction"] for row in rows])
                >= MIN_POSITIVE_JACOBIAN_FRACTION
            )
        ),
        "reverse_positive_jacobian": float(
            np.mean(
                np.asarray([row["reverse_positive_jacobian_fraction"] for row in rows])
                >= MIN_POSITIVE_JACOBIAN_FRACTION
            )
        ),
        "target_cycle": float(
            np.mean(np.asarray([row["target_cycle_mean"] for row in rows]) <= MAX_CYCLE_RESIDUAL)
        ),
        "source_cycle": float(
            np.mean(np.asarray([row["source_cycle_mean"] for row in rows]) <= MAX_CYCLE_RESIDUAL)
        ),
        "target_cycle_valid": float(
            np.mean(
                np.asarray([row["target_cycle_valid_fraction"] for row in rows])
                >= MIN_VALID_FLOW_FRACTION
            )
        ),
        "source_cycle_valid": float(
            np.mean(
                np.asarray([row["source_cycle_valid_fraction"] for row in rows])
                >= MIN_VALID_FLOW_FRACTION
            )
        ),
        "forward_edge_ncc_improved": float(np.mean(forward_edge_gain > 0.0)),
        "reverse_edge_ncc_improved": float(np.mean(reverse_edge_gain > 0.0)),
    }
    return report


def _split_pass(metrics: dict) -> bool:
    rates = metrics["frame_pass_rate"]
    return bool(
        metrics["visible_to_infrared_backward_map"]["frame_pass_rate"]["joint"]
        >= MIN_FRAME_PASS_RATE
        and metrics["infrared_to_visible_backward_map"]["frame_pass_rate"]["joint"]
        >= MIN_FRAME_PASS_RATE
        and all(
            rates[key] >= MIN_FRAME_PASS_RATE
            for key in (
                "forward_valid_flow",
                "reverse_valid_flow",
                "forward_positive_jacobian",
                "reverse_positive_jacobian",
                "target_cycle",
                "source_cycle",
                "target_cycle_valid",
                "source_cycle_valid",
            )
        )
        and metrics["forward_edge_ncc_gain"]["p50"] >= MIN_EDGE_NCC_MEDIAN_GAIN
        and metrics["reverse_edge_ncc_gain"]["p50"] >= MIN_EDGE_NCC_MEDIAN_GAIN
        and rates["forward_edge_ncc_improved"] >= MIN_EDGE_NCC_IMPROVEMENT_RATE
        and rates["reverse_edge_ncc_improved"] >= MIN_EDGE_NCC_IMPROVEMENT_RATE
    )


def gate_report(metrics: dict[str, dict], exhaustive: bool) -> dict[str, str]:
    train_pass = _split_pass(metrics["train"])
    validation_pass = _split_pass(metrics["val"])
    passed = train_pass and validation_pass
    if not exhaustive:
        return {
            "sequence_balanced_screen": "pass" if passed else "hold",
            "held_out_validation": "candidate" if passed else "hold",
            "dense_paired_image_registration": "hold_pending_exhaustive_audit",
            "generator_training_eligible": "hold_pending_exhaustive_audit",
        }
    return {
        "sequence_balanced_screen": "not_applicable",
        "train_fit_sanity": "pass" if train_pass else "hold",
        "held_out_validation": "pass" if validation_pass else "hold",
        "dense_paired_image_registration": "pass" if passed else "hold",
        "generator_training_eligible": "pass" if passed else "hold",
    }


def _checkpoint_provenance(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("aero_registration") if isinstance(payload, dict) else None
    required = {
        "schema_version": 3,
        "kind": "antiuav300_train_only_bidirectional_v3",
        "dataset": "Anti-UAV300",
        "fit_split": "train",
        "validation_or_test_access": "none",
        "epochs": 300,
        "pairs_per_sequence_per_epoch": 16,
        "unique_train_pairs": 141816,
        "batch_size": 8,
        "precision": "float32",
        "seed": 0,
    }
    if not isinstance(metadata, dict) or not all(
        metadata.get(key) == value for key, value in required.items()
    ):
        raise ValueError("checkpoint violates frozen bidirectional v3 protocol")
    return {
        "checkpoint_variant": "Anti-UAV300_train_only_bidirectional_v3",
        "checkpoint_sha256": file_sha256(checkpoint),
        "anti_uav_fitting": "official train split only",
        "training_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.samples_per_sequence < 0 or args.batch_size <= 0:
        parser.error("samples per sequence must be non-negative and batch size must be positive")
    root = args.root.resolve()
    checkpoint = args.checkpoint.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    provenance = _checkpoint_provenance(checkpoint)
    model = load_superfusion_matcher(checkpoint, device)
    rows: dict[str, list[dict]] = defaultdict(list)
    started = time.perf_counter()
    processed = 0
    pairs = _iter_pairs(root, tuple(args.splits), args.samples_per_sequence)
    with torch.inference_mode():
        for batch in _batches(pairs, args.batch_size):
            visible = (
                torch.from_numpy(np.stack([pair.visible for pair in batch]))
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            infrared = (
                torch.from_numpy(np.stack([pair.infrared for pair in batch]))
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            visible_boxes = torch.from_numpy(np.stack([pair.source_box for pair in batch])).to(
                device=device, dtype=torch.float32
            )
            infrared_boxes = torch.from_numpy(np.stack([pair.target_box for pair in batch])).to(
                device=device, dtype=torch.float32
            )
            forward = model(infrared, visible, direction="visible_to_infrared")
            reverse = model(infrared, visible, direction="infrared_to_visible")
            predicted_visible = transform_boxes_target_to_source(infrared_boxes, forward)
            predicted_infrared = transform_boxes_target_to_source(visible_boxes, reverse)
            cycles = inverse_consistency_statistics(forward, reverse)
            forward_registered = warp_source_to_target(visible, forward)
            reverse_registered = warp_source_to_target(infrared, reverse)
            baseline_edge = _edge_ncc(visible, infrared)
            values = {
                "predicted_visible": predicted_visible.cpu().numpy(),
                "predicted_infrared": predicted_infrared.cpu().numpy(),
                "forward_valid": _flow_valid_fraction(forward).cpu().numpy(),
                "reverse_valid": _flow_valid_fraction(reverse).cpu().numpy(),
                "forward_jacobian": _positive_jacobian_fraction(forward).cpu().numpy(),
                "reverse_jacobian": _positive_jacobian_fraction(reverse).cpu().numpy(),
                "forward_edge": (_edge_ncc(forward_registered, infrared) - baseline_edge)
                .cpu()
                .numpy(),
                "reverse_edge": (_edge_ncc(reverse_registered, visible) - baseline_edge)
                .cpu()
                .numpy(),
                **{key: value.cpu().numpy() for key, value in cycles.items()},
            }
            for index, pair in enumerate(batch):
                rows[pair.split].append(
                    {
                        "sequence_id": pair.sequence_id,
                        "frame_index": pair.frame_index,
                        "visible_box": pair.source_box.tolist(),
                        "infrared_box": pair.target_box.tolist(),
                        "predicted_visible_box": values["predicted_visible"][index].tolist(),
                        "predicted_infrared_box": values["predicted_infrared"][index].tolist(),
                        "forward_valid_flow_fraction": float(values["forward_valid"][index]),
                        "reverse_valid_flow_fraction": float(values["reverse_valid"][index]),
                        "forward_positive_jacobian_fraction": float(
                            values["forward_jacobian"][index]
                        ),
                        "reverse_positive_jacobian_fraction": float(
                            values["reverse_jacobian"][index]
                        ),
                        "forward_edge_gain": float(values["forward_edge"][index]),
                        "reverse_edge_gain": float(values["reverse_edge"][index]),
                        **{
                            key: float(values[key][index])
                            for key in (
                                "target_cycle_mean",
                                "target_cycle_max",
                                "source_cycle_mean",
                                "source_cycle_max",
                                "target_cycle_valid_fraction",
                                "source_cycle_valid_fraction",
                            )
                        },
                    }
                )
            processed += len(batch)
            if processed % max(args.batch_size * 20, args.batch_size) == 0:
                elapsed = time.perf_counter() - started
                print(f"processed {processed} pairs ({processed / elapsed:.1f}/s)", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = {split: _split_report(rows[split]) for split in args.splits}
    exhaustive = args.samples_per_sequence == 0
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_bidirectional_dense_registration_audit_v3",
        "root": str(root),
        "model": provenance,
        "data_usage": {
            "evaluated_splits": list(args.splits),
            "frame_selection": (
                "all usable paired target frames"
                if exhaustive
                else f"{args.samples_per_sequence} endpoint-inclusive usable pairs per sequence"
            ),
            "test_access": "none",
        },
        "coordinate_semantics": {
            "visible_to_infrared": (
                "infrared output/target coordinates map directly to visible "
                "input/source coordinates"
            ),
            "infrared_to_visible": (
                "visible output/target coordinates map directly to infrared "
                "input/source coordinates"
            ),
            "box_gate": (
                "each annotation is mapped through the corresponding native backward field; "
                "no box enclosure is numerically inverted"
            ),
        },
        "thresholds": {
            "minimum_bbox_iou": MIN_IOU,
            "maximum_centroid_shift_fraction": MAX_CENTROID_SHIFT_FRACTION,
            "maximum_absolute_area_ratio_change": MAX_AREA_RATIO_CHANGE,
            "minimum_frame_pass_rate": MIN_FRAME_PASS_RATE,
            "minimum_valid_flow_fraction": MIN_VALID_FLOW_FRACTION,
            "maximum_mean_cycle_residual": MAX_CYCLE_RESIDUAL,
            "minimum_positive_jacobian_fraction": MIN_POSITIVE_JACOBIAN_FRACTION,
            "minimum_edge_ncc_median_gain": MIN_EDGE_NCC_MEDIAN_GAIN,
            "minimum_edge_ncc_improvement_rate": MIN_EDGE_NCC_IMPROVEMENT_RATE,
        },
        "metrics": metrics,
        "gates": gate_report(metrics, exhaustive),
        "runtime": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "elapsed_seconds": elapsed,
            "pairs_per_second": processed / elapsed,
        },
    }
    report["bidirectional_registration_audit_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gates": report["gates"], "metrics": metrics}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
