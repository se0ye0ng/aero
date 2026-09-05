"""COCO metrics plus the effect size the hypotheses are actually about.

Reporting raw AP per arm is not enough: every hypothesis concerns ``dAP``, the change caused
by adding generated data at a matched budget. Small samples return a screening estimate only;
adequately sized matched samples may also return a paired bootstrap interval.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np


def _load_coco_ground_truth(ground_truth):
    try:
        from pycocotools.coco import COCO
    except ImportError as error:  # pragma: no cover - exercised in minimal installations
        raise RuntimeError("COCO evaluation requires the 'detect' extra") from error

    if isinstance(ground_truth, COCO):
        return ground_truth
    if isinstance(ground_truth, (str, Path)):
        with contextlib.redirect_stdout(io.StringIO()):
            return COCO(str(ground_truth))
    if isinstance(ground_truth, dict):
        coco = COCO()
        coco.dataset = copy.deepcopy(ground_truth)
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
        return coco
    raise TypeError("ground_truth must be a COCO object, JSON path, or COCO-format dictionary")


def _load_predictions(predictions) -> list[dict]:
    if isinstance(predictions, (str, Path)):
        predictions = json.loads(Path(predictions).read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise TypeError("predictions must be a list or a JSON path containing a list")
    required = {"image_id", "category_id", "bbox", "score"}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or not required <= prediction.keys():
            raise ValueError(f"prediction {index} must contain {sorted(required)}")
        box = prediction["bbox"]
        if len(box) != 4 or not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"prediction {index} has an invalid COCO xywh box")
        if not np.isfinite(float(prediction["score"])):
            raise ValueError(f"prediction {index} has a non-finite score")
    return predictions


def _empty_results(coco_gt):
    """Construct a COCO result object because older pycocotools rejects an empty list."""
    from pycocotools.coco import COCO

    result = COCO()
    result.dataset = {
        "images": copy.deepcopy(coco_gt.dataset.get("images", [])),
        "categories": copy.deepcopy(coco_gt.dataset.get("categories", [])),
        "annotations": [],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        result.createIndex()
    return result


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else 0.0


def coco_metrics(predictions, ground_truth) -> dict[str, object]:
    """Evaluate standard COCO boxes overall and per class.

    Direct comparisons require the same split, category mapping and pinned evaluator version.
    Predictions use ordinary COCO result dictionaries with ``bbox`` in xywh format.
    """
    try:
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:  # pragma: no cover - exercised in minimal installations
        raise RuntimeError("COCO evaluation requires the 'detect' extra") from error

    coco_gt = _load_coco_ground_truth(ground_truth)
    predictions = _load_predictions(predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions) if predictions else _empty_results(coco_gt)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(coco_gt.getImgIds())
        evaluator.params.catIds = sorted(coco_gt.getCatIds())
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    # precision: [IoU, recall threshold, category, area, max detections]
    # recall:    [IoU, category, area, max detections]
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    area_all = evaluator.params.areaRngLbl.index("all")
    max_det_100 = evaluator.params.maxDets.index(100)
    iou_50 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.5)))

    categories = {category["id"]: category["name"] for category in coco_gt.dataset["categories"]}
    per_class: dict[str, dict[str, float | int]] = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        per_class[categories[category_id]] = {
            "category_id": int(category_id),
            "ap_50_95": _mean_valid(precision[:, :, category_index, area_all, max_det_100]),
            "ap_50": _mean_valid(precision[iou_50, :, category_index, area_all, max_det_100]),
            "ar_50_95": _mean_valid(recall[:, category_index, area_all, max_det_100]),
            "ar_50": _mean_valid(recall[iou_50, category_index, area_all, max_det_100]),
        }

    return {
        "evaluator": "pycocotools.COCOeval",
        "iou_type": "bbox",
        "images": len(evaluator.params.imgIds),
        "detections": len(predictions),
        "map_50_95": float(evaluator.stats[0]),
        "map_50": float(evaluator.stats[1]),
        "mar_50_95": float(evaluator.stats[8]),
        "mar_50": _mean_valid(recall[iou_50, :, area_all, max_det_100]),
        "per_class": per_class,
    }


def delta_ap(
    ap_with_gen: np.ndarray,
    ap_real_only: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    min_bootstrap_seeds: int = 10,
) -> dict[str, float | int | bool | str | None]:
    """Effect size and, when powered, a paired bootstrap interval over matched seeds."""
    a = np.asarray(ap_with_gen, dtype=np.float64)
    b = np.asarray(ap_real_only, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size or a.size == 0:
        raise ValueError("AP arrays must be non-empty 1-D arrays matched by seed")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if min_bootstrap_seeds < 2:
        raise ValueError("min_bootstrap_seeds must be at least 2")

    differences = a - b
    point = float(differences.mean())
    if differences.size < min_bootstrap_seeds:
        return {
            "delta_ap": point,
            "ci_low": None,
            "ci_high": None,
            "sign_consistent": None,
            "n_seeds": int(differences.size),
            "inference": "screening_only",
        }

    rng = np.random.default_rng(seed)
    boots = rng.choice(differences, size=(n_boot, differences.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta_ap": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "sign_consistent": bool(lo > 0 or hi < 0),
        "n_seeds": int(differences.size),
        "inference": "paired_percentile_bootstrap",
    }
