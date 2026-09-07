"""YOLOX evaluator that persists predictions and the complete pinned COCO metric set."""

from __future__ import annotations

import json
import math
from pathlib import Path

from yolox.evaluators import COCOEvaluator
from yolox.utils import is_main_process

from aero_ir.detect.evaluate import coco_metrics
from aero_ir.utils.manifest import canonical_hash, file_sha256


def filter_invalid_model_predictions(predictions: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Drop numerical degeneracies that cannot be represented as COCO detections."""
    valid: list[dict] = []
    counts = {
        "received": len(predictions),
        "retained": 0,
        "dropped_non_finite": 0,
        "dropped_non_positive_extent": 0,
    }
    required = {"image_id", "category_id", "bbox", "score"}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or not required <= prediction.keys():
            raise ValueError(f"model prediction {index} is not a COCO result dictionary")
        box = prediction["bbox"]
        if not isinstance(box, list | tuple) or len(box) != 4:
            raise ValueError(f"model prediction {index} does not contain a four-value box")
        try:
            coordinates = [float(value) for value in box]
            score = float(prediction["score"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"model prediction {index} contains non-numeric values") from error
        if not all(math.isfinite(value) for value in (*coordinates, score)):
            counts["dropped_non_finite"] += 1
            continue
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            counts["dropped_non_positive_extent"] += 1
            continue
        valid.append(prediction)
    counts["retained"] = len(valid)
    return valid, counts


class CompleteCOCOEvaluator(COCOEvaluator):
    """Persist evaluator inputs/outputs instead of returning only YOLOX's two AP values."""

    def __init__(self, *args, metrics_path: str | Path, predictions_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_path = Path(metrics_path)
        self.predictions_path = Path(predictions_path)

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0.0, 0.0, None

        data_dict, prediction_filter = filter_invalid_model_predictions(data_dict)
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions_path.write_text(
            json.dumps(data_dict, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = coco_metrics(data_dict, self.dataloader.dataset.coco)
        inference_time = float(statistics[0].item())
        nms_time = float(statistics[1].item())
        n_samples = max(float(statistics[2].item()), 1.0)
        denominator = n_samples * int(self.dataloader.batch_size)
        payload = {
            "schema_version": 1,
            "kind": "flir_yolox_coco_metrics",
            "status": "pass",
            **metrics,
            "prediction_filter": prediction_filter,
            "timing": {
                "forward_ms_per_image": 1000.0 * inference_time / denominator,
                "nms_ms_per_image": 1000.0 * nms_time / denominator,
                "inference_ms_per_image": 1000.0 * (inference_time + nms_time) / denominator,
            },
            "predictions": {
                "path": str(self.predictions_path),
                "sha256": file_sha256(self.predictions_path),
            },
        }
        payload["metrics_sha256"] = canonical_hash(payload)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = (
            "AERO prediction filtering:\n"
            + json.dumps(prediction_filter, indent=2, sort_keys=True)
            + "\nAERO complete COCO metrics:\n"
            + json.dumps(metrics, indent=2, sort_keys=True)
        )
        return float(metrics["map_50_95"]), float(metrics["map_50"]), summary
