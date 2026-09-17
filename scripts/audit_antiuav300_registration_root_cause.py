#!/usr/bin/env python3
"""Diagnose direction, inversion and topology failures in Anti-UAV300 registration."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.superfusion import (
    box_perimeter_points,
    load_superfusion_matcher,
    sample_target_to_source_jacobian,
    transform_boxes_source_to_target,
    transform_boxes_source_to_target_newton,
    transform_boxes_target_to_source,
    warp_source_to_target,
)
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import (
    _batches,
    _bbox_metrics,
    _iter_pairs,
)

SCHEMA_VERSION = 1


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


def _path_metrics(rows: list[dict], prediction: str, target: str) -> dict:
    predicted = np.asarray([row[prediction] for row in rows])
    expected = np.asarray([row[target] for row in rows])
    metrics = _bbox_metrics(predicted, expected)
    return {
        "bbox_iou": _summary(metrics["bbox_iou"]),
        "centroid_shift_fraction": _summary(metrics["centroid_shift_fraction"]),
        "absolute_area_ratio_change": _summary(metrics["absolute_area_ratio_change"]),
        "frame_pass_rate": {
            "iou": float(np.mean(metrics["bbox_iou"] >= 0.6)),
            "centroid_shift": float(np.mean(metrics["centroid_shift_fraction"] <= 0.25)),
            "area": float(np.mean(metrics["absolute_area_ratio_change"] <= 0.5)),
            "in_bounds": float(np.mean(metrics["in_bounds"])),
            "joint": float(np.mean(metrics["joint"])),
        },
    }


def _draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    height, width = image.shape[:2]
    center = box[:2] * np.asarray([width, height])
    size = box[2:] * np.asarray([width, height])
    lower = np.rint(center - size / 2).astype(int)
    upper = np.rint(center + size / 2).astype(int)
    cv2.rectangle(image, tuple(lower), tuple(upper), color, 2)
    cv2.putText(
        image,
        label,
        (int(lower[0]), max(12, int(lower[1]) - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def _determinant_map(displacement: torch.Tensor) -> torch.Tensor:
    height, width = displacement.shape[2:]
    derivative_x = (displacement[:, :, :, 1:] - displacement[:, :, :, :-1]) * ((width - 1) / 2.0)
    derivative_y = (displacement[:, :, 1:, :] - displacement[:, :, :-1, :]) * ((height - 1) / 2.0)
    return (1.0 + derivative_x[:, 0, :-1]) * (1.0 + derivative_y[:, 1, :, :-1]) - derivative_x[
        :, 1, :-1
    ] * derivative_y[:, 0, :, :-1]


def _save_overlay(
    path: Path,
    visible: np.ndarray,
    infrared: np.ndarray,
    registered: np.ndarray,
    determinant: np.ndarray,
    row: dict,
) -> None:
    source_panel = visible.copy()
    _draw_box(source_panel, np.asarray(row["source_box"]), (0, 255, 0), "source")
    _draw_box(
        source_panel,
        np.asarray(row["direct_predicted_source_box"]),
        (255, 0, 255),
        "direct",
    )
    target_panel = infrared.copy()
    _draw_box(target_panel, np.asarray(row["target_box"]), (0, 255, 0), "target")
    _draw_box(
        target_panel,
        np.asarray(row["legacy_predicted_target_box"]),
        (255, 0, 0),
        "fixed",
    )
    _draw_box(
        target_panel,
        np.asarray(row["hybrid_inverse_predicted_target_box"]),
        (0, 255, 255),
        "hybrid",
    )
    registered_panel = registered.copy()
    _draw_box(registered_panel, np.asarray(row["target_box"]), (0, 255, 0), "target")
    clipped = np.clip(determinant, 0.0, 2.0)
    heat = cv2.applyColorMap(np.uint8(clipped / 2.0 * 255.0), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(cv2.resize(heat, (256, 256)), cv2.COLOR_BGR2RGB)
    cv2.putText(
        heat,
        f"det<=0 {row['full_field_fold_fraction']:.1%}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    canvas = np.concatenate((source_panel, target_panel, registered_panel, heat), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write overlay {path}")


def _affine_field(
    scale: float,
    translation: tuple[float, float],
    *,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, device=device)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack(
        (
            (scale - 1.0) * grid_x + translation[0],
            (scale - 1.0) * grid_y + translation[1],
        )
    ).unsqueeze(0)


def _synthetic_audit(device: torch.device) -> dict:
    cases = []
    for scale, translation in ((1.0, (0.12, -0.08)), (2.1, (0.02, -0.04))):
        target = torch.tensor([[0.48, 0.52, 0.08, 0.06]], device=device)
        displacement = _affine_field(scale, translation, size=64, device=device)
        source = transform_boxes_target_to_source(target, displacement)
        fixed, fixed_residual = transform_boxes_source_to_target(source, displacement)
        newton, newton_residual, determinant = transform_boxes_source_to_target_newton(
            source, displacement, inverse_iterations=20
        )
        fixed_iou = _bbox_metrics(fixed.cpu().numpy(), target.cpu().numpy())["bbox_iou"][0]
        newton_iou = _bbox_metrics(newton.cpu().numpy(), target.cpu().numpy())["bbox_iou"][0]
        cases.append(
            {
                "scale": scale,
                "translation": list(translation),
                "fixed_point_iou": float(fixed_iou),
                "fixed_point_residual": float(fixed_residual.item()),
                "newton_iou": float(newton_iou),
                "newton_residual": float(newton_residual.item()),
                "minimum_map_jacobian_determinant": float(determinant.min()),
            }
        )
    return {"cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--failed-screen", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overlay-dir", required=True, type=Path)
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overlays-per-split", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.samples_per_sequence, args.batch_size, args.overlays_per_split) <= 0:
        parser.error("sample, batch and overlay counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    failed_screen = json.loads(args.failed_screen.read_text(encoding="utf-8"))
    unhashed = {
        key: value
        for key, value in failed_screen.items()
        if key != "dense_registration_audit_sha256"
    }
    if canonical_hash(unhashed) != failed_screen.get("dense_registration_audit_sha256"):
        raise ValueError("failed screen content hash does not verify")
    worst_keys = {
        split: {
            (item["sequence_id"], int(item["frame_index"]))
            for item in failed_screen["metrics"][split]["worst_bbox_iou_examples"][
                : args.overlays_per_split
            ]
        }
        for split in ("train", "val")
    }

    model = load_superfusion_matcher(args.checkpoint, device)
    rows: dict[str, list[dict]] = defaultdict(list)
    overlays: list[dict] = []
    started = time.perf_counter()
    pairs = _iter_pairs(args.root.resolve(), ("train", "val"), args.samples_per_sequence)
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
            source_boxes = torch.from_numpy(np.stack([pair.source_box for pair in batch])).to(
                device=device, dtype=torch.float32
            )
            target_boxes = torch.from_numpy(np.stack([pair.target_box for pair in batch])).to(
                device=device, dtype=torch.float32
            )
            displacement = model(infrared, visible, direction="visible_to_infrared")
            direct = transform_boxes_target_to_source(target_boxes, displacement)
            fixed, fixed_residual = transform_boxes_source_to_target(source_boxes, displacement)
            hybrid, hybrid_residual, hybrid_determinant = transform_boxes_source_to_target_newton(
                source_boxes, displacement, inverse_iterations=20
            )
            target_points = box_perimeter_points(target_boxes)
            target_jacobian = sample_target_to_source_jacobian(target_points, displacement)
            target_determinant = torch.linalg.det(target_jacobian)
            identity = torch.eye(2, device=device, dtype=target_jacobian.dtype)
            displacement_spectral_norm = torch.linalg.matrix_norm(target_jacobian - identity, ord=2)
            determinant_map = _determinant_map(displacement)
            registered = warp_source_to_target(visible, displacement).clamp(0.0, 1.0)
            arrays = {
                "direct": direct.cpu().numpy(),
                "fixed": fixed.cpu().numpy(),
                "hybrid": hybrid.cpu().numpy(),
                "fixed_residual": fixed_residual.cpu().numpy(),
                "hybrid_residual": hybrid_residual.cpu().numpy(),
                "hybrid_determinant": hybrid_determinant.cpu().numpy(),
                "target_determinant": target_determinant.cpu().numpy(),
                "spectral": displacement_spectral_norm.cpu().numpy(),
                "determinant_map": determinant_map.cpu().numpy(),
                "registered": np.uint8(registered.permute(0, 2, 3, 1).cpu().numpy() * 255.0),
            }
            for index, pair in enumerate(batch):
                row = {
                    "split": pair.split,
                    "sequence_id": pair.sequence_id,
                    "frame_index": pair.frame_index,
                    "source_box": pair.source_box.tolist(),
                    "target_box": pair.target_box.tolist(),
                    "direct_predicted_source_box": arrays["direct"][index].tolist(),
                    "legacy_predicted_target_box": arrays["fixed"][index].tolist(),
                    "hybrid_inverse_predicted_target_box": arrays["hybrid"][index].tolist(),
                    "legacy_inverse_residual": float(arrays["fixed_residual"][index]),
                    "hybrid_inverse_residual": float(arrays["hybrid_residual"][index]),
                    "target_perimeter_fold_fraction": float(
                        np.mean(arrays["target_determinant"][index] <= 0.0)
                    ),
                    "hybrid_inverse_perimeter_fold_fraction": float(
                        np.mean(arrays["hybrid_determinant"][index] <= 0.0)
                    ),
                    "target_perimeter_noncontractive_fraction": float(
                        np.mean(arrays["spectral"][index] >= 1.0)
                    ),
                    "full_field_fold_fraction": float(
                        np.mean(arrays["determinant_map"][index] <= 0.0)
                    ),
                }
                rows[pair.split].append(row)
                key = (pair.sequence_id, pair.frame_index)
                if key in worst_keys[pair.split]:
                    overlay_path = (
                        args.overlay_dir
                        / pair.split
                        / f"{pair.sequence_id}__{pair.frame_index:06d}.png"
                    )
                    _save_overlay(
                        overlay_path,
                        pair.visible,
                        pair.infrared,
                        arrays["registered"][index],
                        arrays["determinant_map"][index],
                        row,
                    )
                    overlays.append(
                        {
                            "split": pair.split,
                            "sequence_id": pair.sequence_id,
                            "frame_index": pair.frame_index,
                            "path": str(overlay_path.resolve()),
                            "sha256": file_sha256(overlay_path),
                        }
                    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    metrics = {}
    for split, split_rows in rows.items():
        metrics[split] = {
            "pairs": len(split_rows),
            "direct_target_to_source": _path_metrics(
                split_rows, "direct_predicted_source_box", "source_box"
            ),
            "legacy_source_to_target": _path_metrics(
                split_rows, "legacy_predicted_target_box", "target_box"
            ),
            "hybrid_source_to_target_diagnostic": _path_metrics(
                split_rows, "hybrid_inverse_predicted_target_box", "target_box"
            ),
            "legacy_inverse_residual": _summary(
                np.asarray([row["legacy_inverse_residual"] for row in split_rows])
            ),
            "hybrid_inverse_residual": _summary(
                np.asarray([row["hybrid_inverse_residual"] for row in split_rows])
            ),
            "target_perimeter_fold_fraction": _summary(
                np.asarray([row["target_perimeter_fold_fraction"] for row in split_rows])
            ),
            "hybrid_inverse_perimeter_fold_fraction": _summary(
                np.asarray([row["hybrid_inverse_perimeter_fold_fraction"] for row in split_rows])
            ),
            "target_perimeter_noncontractive_fraction": _summary(
                np.asarray([row["target_perimeter_noncontractive_fraction"] for row in split_rows])
            ),
            "full_field_fold_fraction": _summary(
                np.asarray([row["full_field_fold_fraction"] for row in split_rows])
            ),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "antiuav300_registration_root_cause_audit",
        "dataset_root": str(args.root.resolve()),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": file_sha256(args.checkpoint),
        },
        "failed_screen": {
            "path": str(args.failed_screen.resolve()),
            "sha256": file_sha256(args.failed_screen),
            "content_sha256": failed_screen["dense_registration_audit_sha256"],
        },
        "selection": f"{args.samples_per_sequence} endpoint-inclusive pairs per sequence",
        "synthetic_known_transforms": _synthetic_audit(device),
        "metrics": metrics,
        "overlays": sorted(
            overlays, key=lambda item: (item["split"], item["sequence_id"], item["frame_index"])
        ),
        "runtime": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "elapsed_seconds": elapsed,
            "pairs_per_second": sum(len(value) for value in rows.values()) / elapsed,
        },
        "interpretation_policy": {
            "inverse_box_paths": (
                "diagnostic only: nonlinear map inversion and axis-aligned box enclosure do not "
                "commute"
            ),
            "noncontractive_flow": "displacement-Jacobian spectral norm is at least one",
            "folded_flow": "target-to-source Jacobian determinant is non-positive",
            "model_error": "direct target-to-source box mapping misses source annotation",
        },
        "conclusion": {
            "primary_screen_error": (
                "the immutable screen inverted an enclosed source box even though the matcher "
                "and v2 loss define a native target-to-source backward sampling map"
            ),
            "qualification_path": (
                "score each native backward map directly and use a separately predicted "
                "reciprocal map for the opposite direction"
            ),
            "v2_status": (
                "remains HOLD because the corrected validation joint pass rate is below 95%"
            ),
        },
    }
    report["root_cause_audit_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "runtime": report["runtime"]}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
