import json
from types import SimpleNamespace

import pytest
import torch
from pycocotools.coco import COCO

from aero_ir.detect.yolox_evaluator import CompleteCOCOEvaluator
from aero_ir.utils.manifest import canonical_hash, file_sha256


def _ground_truth():
    coco = COCO()
    coco.dataset = {
        "images": [{"id": 1, "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [2, 3, 8, 6],
                "area": 48,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 7, "name": "target"}],
    }
    coco.createIndex()
    return coco


def test_complete_evaluator_persists_predictions_and_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.json"
    dataloader = SimpleNamespace(
        dataset=SimpleNamespace(coco=_ground_truth()),
        batch_size=1,
    )
    evaluator = CompleteCOCOEvaluator(
        dataloader=dataloader,
        img_size=(640, 640),
        confthre=0.01,
        nmsthre=0.65,
        num_classes=1,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
    predictions = [{"image_id": 1, "category_id": 7, "bbox": [2, 3, 8, 6], "score": 0.99}]

    ap, ap50, summary = evaluator.evaluate_prediction(
        predictions,
        torch.tensor([0.2, 0.1, 2.0]),
    )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert ap == pytest.approx(1.0)
    assert ap50 == pytest.approx(1.0)
    assert payload["mar_50"] == pytest.approx(1.0)
    assert payload["per_class"]["target"]["ap_50_95"] == pytest.approx(1.0)
    assert payload["timing"]["inference_ms_per_image"] == pytest.approx(150.0)
    assert payload["predictions"]["sha256"] == file_sha256(predictions_path)
    unsigned = {key: value for key, value in payload.items() if key != "metrics_sha256"}
    assert payload["metrics_sha256"] == canonical_hash(unsigned)
    assert "AERO complete COCO metrics" in summary


def test_complete_evaluator_filters_numerically_invalid_model_boxes(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.json"
    dataloader = SimpleNamespace(
        dataset=SimpleNamespace(coco=_ground_truth()),
        batch_size=1,
    )
    evaluator = CompleteCOCOEvaluator(
        dataloader=dataloader,
        img_size=(640, 640),
        confthre=0.01,
        nmsthre=0.65,
        num_classes=1,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
    predictions = [
        {"image_id": 1, "category_id": 7, "bbox": [2, 3, 8, 6], "score": 0.99},
        {"image_id": 1, "category_id": 7, "bbox": [2, 3, 0, 6], "score": 0.80},
        {"image_id": 1, "category_id": 7, "bbox": [2, 3, 8, 6], "score": float("nan")},
    ]

    ap, _, summary = evaluator.evaluate_prediction(
        predictions,
        torch.tensor([0.2, 0.1, 2.0]),
    )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    persisted = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert ap == pytest.approx(1.0)
    assert persisted == [{"image_id": 1, "category_id": 7, "bbox": [2, 3, 8, 6], "score": 0.99}]
    assert payload["detections"] == 1
    assert payload["prediction_filter"] == {
        "received": 3,
        "retained": 1,
        "dropped_non_finite": 1,
        "dropped_non_positive_extent": 1,
    }
    assert "AERO prediction filtering" in summary
