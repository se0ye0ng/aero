"""YOLOX evaluator that persists predictions and the complete pinned COCO metric set."""

from __future__ import annotations

import json
from pathlib import Path

from yolox.evaluators import COCOEvaluator
from yolox.utils import is_main_process

from aero_ir.detect.evaluate import coco_metrics
from aero_ir.utils.manifest import canonical_hash, file_sha256


class CompleteCOCOEvaluator(COCOEvaluator):
    """Persist evaluator inputs/outputs instead of returning only YOLOX's two AP values."""

    def __init__(self, *args, metrics_path: str | Path, predictions_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_path = Path(metrics_path)
        self.predictions_path = Path(predictions_path)

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0.0, 0.0, None

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
        summary = "AERO complete COCO metrics:\n" + json.dumps(metrics, indent=2, sort_keys=True)
        return float(metrics["map_50_95"]), float(metrics["map_50"]), summary
