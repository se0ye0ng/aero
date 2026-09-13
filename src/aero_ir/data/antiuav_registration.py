"""Train-only Anti-UAV300 RGB-to-IR target-box registration audit."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aero_ir.data.antiuav import ANTIUAV300_SPLITS, load_split_manifest
from aero_ir.utils.manifest import canonical_hash, file_sha256

REGISTRATION_AUDIT_SCHEMA = 1
VISIBLE_SIZE = (1920.0, 1080.0)
INFRARED_SIZE = (640.0, 512.0)
DEFAULT_MIN_IOU = 0.6
DEFAULT_MAX_CENTROID_SHIFT_FRACTION = 0.25
DEFAULT_MAX_AREA_RATIO_CHANGE = 0.5
DEFAULT_MIN_FRAME_PASS_RATE = 0.95


@dataclass(frozen=True)
class _SplitGeometry:
    source: np.ndarray
    target: np.ndarray
    sequence_ids: np.ndarray
    counts: dict[str, int]
    provenance: dict


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _normalised_box(box: object, size: tuple[float, float]) -> np.ndarray | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    values = (x, y, width, height)
    if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
        return None
    image_width, image_height = size
    if x < 0 or y < 0 or x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
        return None
    return np.asarray(
        [
            (x + width / 2.0) / image_width,
            (y + height / 2.0) / image_height,
            width / image_width,
            height / image_height,
        ],
        dtype=np.float64,
    )


def _collect_split(root: Path, split: str) -> _SplitGeometry:
    manifest_path = root / "label_new" / f"{split}.json"
    sequences = load_split_manifest(root, split)
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    sequence_ids: list[str] = []
    counts: Counter[str] = Counter()
    annotation_inputs = []

    for sequence in sorted(sequences):
        visible_path = root / split / sequence / "visible.json"
        infrared_path = root / split / sequence / "infrared.json"
        visible = _load_json(visible_path)
        infrared = _load_json(infrared_path)
        arrays = (
            visible.get("exist"),
            visible.get("gt_rect"),
            infrared.get("exist"),
            infrared.get("gt_rect"),
        )
        if not all(isinstance(values, list) for values in arrays):
            raise ValueError(f"invalid paired annotation arrays in {split}/{sequence}")
        lengths = {len(values) for values in arrays}
        if len(lengths) != 1:
            raise ValueError(f"paired annotation length mismatch in {split}/{sequence}")
        annotation_inputs.append(
            {
                "sequence_id": sequence,
                "visible_sha256": file_sha256(visible_path),
                "infrared_sha256": file_sha256(infrared_path),
            }
        )

        visible_exist, visible_boxes, infrared_exist, infrared_boxes = arrays
        for vis_present, vis_box, ir_present, ir_box in zip(
            visible_exist,
            visible_boxes,
            infrared_exist,
            infrared_boxes,
            strict=True,
        ):
            counts["paired_frames"] += 1
            if vis_present not in (0, 1) or ir_present not in (0, 1):
                counts["invalid_existence"] += 1
                continue
            if vis_present != ir_present:
                counts["presence_mismatch"] += 1
                continue
            if vis_present == 0:
                counts["both_absent"] += 1
                continue
            normalised_visible = _normalised_box(vis_box, VISIBLE_SIZE)
            normalised_infrared = _normalised_box(ir_box, INFRARED_SIZE)
            if normalised_visible is None:
                counts["invalid_visible_box"] += 1
                continue
            if normalised_infrared is None:
                counts["invalid_infrared_box"] += 1
                continue
            source.append(normalised_visible)
            target.append(normalised_infrared)
            sequence_ids.append(sequence)
            counts["usable_pairs"] += 1

    if not source:
        raise ValueError(f"no usable paired target boxes in Anti-UAV300/{split}")
    return _SplitGeometry(
        source=np.stack(source),
        target=np.stack(target),
        sequence_ids=np.asarray(sequence_ids),
        counts=dict(counts),
        provenance={
            "split_manifest_sha256": file_sha256(manifest_path),
            "annotation_set_sha256": canonical_hash(annotation_inputs),
            "sequences": len(sequences),
            "annotation_files": len(annotation_inputs) * 2,
        },
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    cutoff = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, cutoff, side="left")), len(values) - 1)
    return float(ordered_values[index])


def _sequence_balanced_weights(sequence_ids: np.ndarray) -> np.ndarray:
    names, inverse, counts = np.unique(sequence_ids, return_inverse=True, return_counts=True)
    if not len(names):
        raise ValueError("registration fit has no sequences")
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.mean()


def _robust_fit(
    features: np.ndarray,
    targets: np.ndarray,
    base_weights: np.ndarray,
    *,
    iterations: int = 12,
) -> np.ndarray:
    weights = base_weights.copy()
    coefficients = np.zeros((features.shape[1], targets.shape[1]), dtype=np.float64)
    for _ in range(iterations):
        scale_weights = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(
            features * scale_weights,
            targets * scale_weights,
            rcond=None,
        )[0]
        residuals = np.linalg.norm(targets - features @ coefficients, axis=1)
        scale = max(_weighted_quantile(residuals, base_weights, 0.5) / 0.67448975, 1e-12)
        cutoff = 1.345 * scale
        robust_weights = np.minimum(1.0, cutoff / np.maximum(residuals, 1e-12))
        weights = base_weights * robust_weights
    return coefficients


def _fit_model(training: _SplitGeometry) -> dict:
    source = training.source
    weights = _sequence_balanced_weights(training.sequence_ids)
    center_features = np.column_stack((source[:, 0], source[:, 1], np.ones(len(source))))
    size_features = np.column_stack(
        (
            source[:, 0],
            source[:, 1],
            np.log(source[:, 2]),
            np.log(source[:, 3]),
            np.ones(len(source)),
        )
    )
    center_coefficients = _robust_fit(
        center_features,
        training.target[:, :2],
        weights,
    )
    log_size_coefficients = _robust_fit(
        size_features,
        np.log(training.target[:, 2:]),
        weights,
    )
    return {
        "kind": "sequence_balanced_huber_affine_center_log_linear_size",
        "fit_split": "train",
        "huber_delta": 1.345,
        "iterations": 12,
        "center_feature_order": ["visible_cx", "visible_cy", "intercept"],
        "center_coefficients": center_coefficients.tolist(),
        "log_size_feature_order": [
            "visible_cx",
            "visible_cy",
            "log_visible_width",
            "log_visible_height",
            "intercept",
        ],
        "log_size_coefficients": log_size_coefficients.tolist(),
        "sequence_weighting": "each training sequence has equal total initial weight",
    }


def _apply_model(source: np.ndarray, model: dict) -> np.ndarray:
    center_features = np.column_stack((source[:, 0], source[:, 1], np.ones(len(source))))
    size_features = np.column_stack(
        (
            source[:, 0],
            source[:, 1],
            np.log(source[:, 2]),
            np.log(source[:, 3]),
            np.ones(len(source)),
        )
    )
    center = center_features @ np.asarray(model["center_coefficients"], dtype=np.float64)
    log_size = size_features @ np.asarray(model["log_size_coefficients"], dtype=np.float64)
    size = np.exp(np.clip(log_size, -30.0, 30.0))
    return np.column_stack((center, size))


def _bbox_iou(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted_min = predicted[:, :2] - predicted[:, 2:] / 2.0
    predicted_max = predicted[:, :2] + predicted[:, 2:] / 2.0
    target_min = target[:, :2] - target[:, 2:] / 2.0
    target_max = target[:, :2] + target[:, 2:] / 2.0
    intersection_size = np.maximum(
        0.0,
        np.minimum(predicted_max, target_max) - np.maximum(predicted_min, target_min),
    )
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    union = predicted[:, 2] * predicted[:, 3] + target[:, 2] * target[:, 3] - intersection
    return intersection / np.maximum(union, 1e-12)


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


def _evaluate(
    geometry: _SplitGeometry,
    model: dict,
    *,
    min_iou: float,
    max_centroid_shift_fraction: float,
    max_area_ratio_change: float,
) -> dict:
    predicted = _apply_model(geometry.source, model)
    target = geometry.target
    center_error = np.linalg.norm(predicted[:, :2] - target[:, :2], axis=1)
    target_diagonal = np.linalg.norm(target[:, 2:], axis=1)
    shift_fraction = center_error / np.maximum(target_diagonal, 1e-12)
    iou = _bbox_iou(predicted, target)
    predicted_area = predicted[:, 2] * predicted[:, 3]
    target_area = target[:, 2] * target[:, 3]
    area_ratio_change = np.abs(predicted_area / np.maximum(target_area, 1e-12) - 1.0)
    predicted_min = predicted[:, :2] - predicted[:, 2:] / 2.0
    predicted_max = predicted[:, :2] + predicted[:, 2:] / 2.0
    in_bounds = np.all(predicted_min >= 0.0, axis=1) & np.all(predicted_max <= 1.0, axis=1)
    iou_pass = iou >= min_iou
    shift_pass = shift_fraction <= max_centroid_shift_fraction
    area_pass = area_ratio_change <= max_area_ratio_change
    joint_pass = iou_pass & shift_pass & area_pass & in_bounds

    per_sequence_joint = []
    for sequence in np.unique(geometry.sequence_ids):
        mask = geometry.sequence_ids == sequence
        per_sequence_joint.append(float(np.mean(joint_pass[mask])))
    return {
        "counts": geometry.counts,
        "normalised_center_error": _summary(center_error),
        "centroid_shift_fraction": _summary(shift_fraction),
        "bbox_iou": _summary(iou),
        "absolute_area_ratio_change": _summary(area_ratio_change),
        "frame_pass_rate": {
            "iou": float(np.mean(iou_pass)),
            "centroid_shift": float(np.mean(shift_pass)),
            "area": float(np.mean(area_pass)),
            "in_bounds": float(np.mean(in_bounds)),
            "joint": float(np.mean(joint_pass)),
        },
        "sequence_macro_joint_pass_rate": _summary(np.asarray(per_sequence_joint)),
    }


def build_registration_audit(
    root: str | Path,
) -> dict:
    """Fit only on train and audit the unchanged transform on official val/test splits."""
    root = Path(root).resolve()
    geometry = {split: _collect_split(root, split) for split in ANTIUAV300_SPLITS}
    model = _fit_model(geometry["train"])
    metrics = {
        split: _evaluate(
            geometry[split],
            model,
            min_iou=DEFAULT_MIN_IOU,
            max_centroid_shift_fraction=DEFAULT_MAX_CENTROID_SHIFT_FRACTION,
            max_area_ratio_change=DEFAULT_MAX_AREA_RATIO_CHANGE,
        )
        for split in ANTIUAV300_SPLITS
    }
    train_pass = metrics["train"]["frame_pass_rate"]["joint"] >= DEFAULT_MIN_FRAME_PASS_RATE
    validation_pass = metrics["val"]["frame_pass_rate"]["joint"] >= DEFAULT_MIN_FRAME_PASS_RATE
    transfer_pass = train_pass and validation_pass
    report = {
        "schema_version": REGISTRATION_AUDIT_SCHEMA,
        "kind": "antiuav300_train_fitted_target_box_registration_audit",
        "dataset": "Anti-UAV300",
        "root": str(root),
        "data_usage": {
            "fit": "train only",
            "qualification": "validation only; never used to fit or select coefficients",
            "test": "report only; never used to fit, select, or qualify the transform",
        },
        "source_provenance": {split: geometry[split].provenance for split in ANTIUAV300_SPLITS},
        "thresholds": {
            "min_bbox_iou": DEFAULT_MIN_IOU,
            "max_centroid_shift_fraction_of_target_diagonal": (DEFAULT_MAX_CENTROID_SHIFT_FRACTION),
            "max_absolute_area_ratio_change": DEFAULT_MAX_AREA_RATIO_CHANGE,
            "min_joint_frame_pass_rate": DEFAULT_MIN_FRAME_PASS_RATE,
        },
        "model": model,
        "metrics": metrics,
        "gates": {
            "train_fit_sanity": "pass" if train_pass else "hold",
            "held_out_validation": "pass" if validation_pass else "hold",
            "calibrated_target_box_transfer": "pass" if transfer_pass else "hold",
            "dense_paired_image_registration": "hold",
            "generator_training_eligible": "hold",
        },
        "notes": [
            "The robust model gives every training sequence equal initial total weight.",
            "Validation and test annotations never influence fitted coefficients.",
            "Target trajectories alone cannot establish dense background/image registration.",
            "A failed frame is excluded; clipping a predicted box cannot rescue registration.",
        ],
    }
    report["registration_audit_sha256"] = canonical_hash(report)
    verify_registration_audit(report)
    return report


def verify_registration_audit(report: dict) -> None:
    """Validate content address and the conservative Anti-UAV300 gate semantics."""
    if report.get("schema_version") != REGISTRATION_AUDIT_SCHEMA:
        raise ValueError("unsupported Anti-UAV300 registration-audit schema")
    recorded = report.get("registration_audit_sha256")
    if not isinstance(recorded, str):
        raise ValueError("Anti-UAV300 registration audit has no content hash")
    unsigned = {key: value for key, value in report.items() if key != "registration_audit_sha256"}
    if canonical_hash(unsigned) != recorded:
        raise ValueError("Anti-UAV300 registration-audit hash mismatch")
    if report.get("dataset") != "Anti-UAV300":
        raise ValueError("Anti-UAV300 registration audit has the wrong dataset identity")
    if report.get("kind") != "antiuav300_train_fitted_target_box_registration_audit":
        raise ValueError("Anti-UAV300 registration audit has the wrong kind")
    usage = report.get("data_usage", {})
    if usage.get("fit") != "train only" or not str(usage.get("qualification", "")).startswith(
        "validation only"
    ):
        raise ValueError("Anti-UAV300 registration audit violates split isolation")
    if not str(usage.get("test", "")).startswith("report only"):
        raise ValueError("Anti-UAV300 test annotations must remain report-only")
    expected_thresholds = {
        "min_bbox_iou": DEFAULT_MIN_IOU,
        "max_centroid_shift_fraction_of_target_diagonal": (DEFAULT_MAX_CENTROID_SHIFT_FRACTION),
        "max_absolute_area_ratio_change": DEFAULT_MAX_AREA_RATIO_CHANGE,
        "min_joint_frame_pass_rate": DEFAULT_MIN_FRAME_PASS_RATE,
    }
    if report.get("thresholds") != expected_thresholds:
        raise ValueError("Anti-UAV300 registration thresholds changed")
    if set(report.get("source_provenance", {})) != set(ANTIUAV300_SPLITS):
        raise ValueError("Anti-UAV300 registration provenance must cover every official split")
    model = report.get("model", {})
    if model.get("fit_split") != "train" or model.get("kind") != (
        "sequence_balanced_huber_affine_center_log_linear_size"
    ):
        raise ValueError("Anti-UAV300 registration model must be fit on train")
    center = np.asarray(model.get("center_coefficients"), dtype=np.float64)
    size = np.asarray(model.get("log_size_coefficients"), dtype=np.float64)
    if center.shape != (3, 2) or size.shape != (5, 2):
        raise ValueError("Anti-UAV300 registration coefficient shape changed")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)):
        raise ValueError("Anti-UAV300 registration coefficients must be finite")
    metrics = report.get("metrics", {})
    if set(metrics) != set(ANTIUAV300_SPLITS):
        raise ValueError("Anti-UAV300 registration metrics must cover every official split")
    try:
        train_joint = float(metrics["train"]["frame_pass_rate"]["joint"])
        validation_joint = float(metrics["val"]["frame_pass_rate"]["joint"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Anti-UAV300 registration pass rates are malformed") from error
    train_gate = "pass" if train_joint >= DEFAULT_MIN_FRAME_PASS_RATE else "hold"
    validation_gate = "pass" if validation_joint >= DEFAULT_MIN_FRAME_PASS_RATE else "hold"
    transfer_gate = "pass" if train_gate == validation_gate == "pass" else "hold"
    expected_gates = {
        "train_fit_sanity": train_gate,
        "held_out_validation": validation_gate,
        "calibrated_target_box_transfer": transfer_gate,
        "dense_paired_image_registration": "hold",
        "generator_training_eligible": "hold",
    }
    if report.get("gates") != expected_gates:
        raise ValueError("Anti-UAV300 registration gates disagree with frozen semantics")
