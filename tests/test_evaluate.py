import numpy as np
import pytest

from aero_ir.detect.evaluate import coco_metrics, delta_ap


def test_delta_ap_bootstraps_matched_seed_differences():
    # Every matched run improves by exactly two points, so pairing removes the large
    # between-seed baseline variance from the interval.
    baseline = np.arange(10, dtype=float) * 100.0
    result = delta_ap(baseline + 2.0, baseline)
    assert result["delta_ap"] == pytest.approx(2.0)
    assert result["ci_low"] == pytest.approx(2.0)
    assert result["ci_high"] == pytest.approx(2.0)
    assert result["sign_consistent"]
    assert result["inference"] == "paired_percentile_bootstrap"


def test_delta_ap_marks_three_seeds_as_screening_only():
    result = delta_ap(np.array([2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0]))
    assert result["delta_ap"] == pytest.approx(1.0)
    assert result["ci_low"] is None
    assert result["ci_high"] is None
    assert result["sign_consistent"] is None
    assert result["inference"] == "screening_only"


def test_delta_ap_requires_matched_nonempty_arrays():
    with pytest.raises(ValueError, match="matched by seed"):
        delta_ap(np.array([1.0]), np.array([1.0, 2.0]))


def test_coco_metrics_scores_a_perfect_box():
    pytest.importorskip("pycocotools")
    ground_truth = {
        "images": [{"id": 1, "width": 20, "height": 20, "file_name": "one.png"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [2, 3, 5, 6],
                "area": 30,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 7, "name": "target"}],
    }
    predictions = [{"image_id": 1, "category_id": 7, "bbox": [2, 3, 5, 6], "score": 0.9}]

    result = coco_metrics(predictions, ground_truth)

    assert result["images"] == 1
    assert result["detections"] == 1
    assert result["map_50_95"] == pytest.approx(1.0)
    assert result["map_50"] == pytest.approx(1.0)
    assert result["mar_50_95"] == pytest.approx(1.0)
    assert result["mar_50"] == pytest.approx(1.0)
    assert result["per_class"]["target"]["ap_50_95"] == pytest.approx(1.0)


def test_coco_metrics_handles_no_detections():
    pytest.importorskip("pycocotools")
    ground_truth = {
        "images": [{"id": 1, "width": 20, "height": 20, "file_name": "one.png"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "target"}],
    }

    result = coco_metrics([], ground_truth)

    assert result["map_50_95"] == 0.0
    assert result["mar_50_95"] == 0.0
