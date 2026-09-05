import json

import numpy as np
from PIL import Image

from aero_ir.data.flir import (
    FLIR_RGB_SPLITS,
    FLIR_SPLITS,
    audit_release,
    build_detector_manifest,
    load_thermal_sample,
    verify_detector_manifest,
)
from aero_ir.data.registry import build_dataset


def _write_release(root):
    filenames = {}
    for index, (split, split_dir) in enumerate(FLIR_SPLITS.items(), start=1):
        video_id = f"video{index}"
        filename = f"video-{video_id}-frame-000001-token.jpg"
        filenames[split] = filename
        data_dir = root / split_dir / "data"
        analytics_dir = root / split_dir / "analyticsData"
        data_dir.mkdir(parents=True)
        analytics_dir.mkdir()
        Image.fromarray(np.full((8, 10), index, dtype=np.uint8)).save(data_dir / filename)
        analytics = np.arange(80, dtype=np.uint16).reshape(8, 10) + 1000 + index
        Image.fromarray(analytics).save(analytics_dir / filename.replace(".jpg", ".tiff"))
        coco = {
            "images": [
                {
                    "id": index,
                    "file_name": f"data/{filename}",
                    "height": 8,
                    "width": 10,
                    "extra_info": {"video_id": video_id},
                }
            ],
            "annotations": [
                {"id": index, "image_id": index, "category_id": 1, "bbox": [1, 1, 3, 4]}
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        (root / split_dir / "coco.json").write_text(json.dumps(coco), encoding="utf-8")

    rgb_names = {}
    for index, (split, split_dir) in enumerate(FLIR_RGB_SPLITS.items(), start=11):
        video_id = f"rgbvideo{index}"
        rgb_name = f"video-{video_id}-frame-000001-token.jpg"
        rgb_names[split] = rgb_name
        rgb_dir = root / split_dir / "data"
        rgb_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(rgb_dir / rgb_name)
        coco = {
            "images": [
                {
                    "id": index,
                    "file_name": f"data/{rgb_name}",
                    "height": 8,
                    "width": 10,
                    "extra_info": {"video_id": video_id},
                }
            ],
            "annotations": [
                {"id": index, "image_id": index, "category_id": 1, "bbox": [1, 1, 3, 4]}
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        (root / split_dir / "coco.json").write_text(json.dumps(coco), encoding="utf-8")

    video_map = root.parent / "rgb_to_thermal_vid_map.json"
    video_map.write_text(json.dumps({rgb_names["test"]: filenames["test"]}), encoding="utf-8")
    return video_map


def test_release_audit_and_sequence_gates(tmp_path):
    root = tmp_path / "FLIR_ADAS_v2"
    video_map = _write_release(root)

    report = audit_release(root, video_map)

    assert report["gates"]["thermal_baseline_data_integrity"] == "pass"
    assert report["gates"]["rgb_source_data_integrity"] == "pass"
    assert report["gates"]["time_synchronised_video_pairs"] == "pass"
    assert report["gates"]["label_transfer_registration"] == "hold"
    assert report["thermal_sequence_overlap"] == {
        "train_val": [],
        "train_test": [],
        "val_test": [],
    }


def test_load_analytics_sample(tmp_path):
    root = tmp_path / "FLIR_ADAS_v2"
    _write_release(root)

    images, boxes, image_ids = load_thermal_sample(
        root, "train", n_images=1, seed=0, representation="analytics16"
    )

    assert images[0].shape == (8, 10)
    assert images[0][0, 0] == 1001
    assert boxes == [[[1.0, 1.0, 3.0, 4.0]]]
    assert image_ids == [1]


def test_detector_manifest_is_deterministic_and_registry_loads_it(tmp_path):
    root = tmp_path / "FLIR_ADAS_v2"
    video_map = _write_release(root)
    audit = audit_release(root, video_map)

    first = build_detector_manifest(root, classes=("person",), audit_report=audit)
    second = build_detector_manifest(root, classes=("person",), audit_report=audit)
    assert first == second
    assert first["splits"]["train"]["counts"] == {
        "images": 1,
        "images_with_selected_annotations": 1,
        "selected_annotations": 1,
        "excluded_annotations": 0,
    }
    verify_detector_manifest(first)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(first), encoding="utf-8")
    dataset = build_dataset(
        {"name": "flir_urban", "root": str(root), "manifest": str(manifest_path)},
        split="train",
    )
    image, target = dataset[0]
    assert image.dtype == np.uint16
    assert image.shape == (8, 10)
    assert target["boxes_xywh"].tolist() == [[1.0, 1.0, 3.0, 4.0]]
    assert target["labels"].tolist() == [0]
    assert target["source_category_ids"].tolist() == [1]


def test_detector_manifest_rejects_tampering(tmp_path):
    root = tmp_path / "FLIR_ADAS_v2"
    video_map = _write_release(root)
    manifest = build_detector_manifest(
        root,
        classes=("person",),
        audit_report=audit_release(root, video_map),
    )
    manifest["splits"]["train"]["records"][0]["width"] = 999

    with np.testing.assert_raises_regex(ValueError, "hash mismatch"):
        verify_detector_manifest(manifest)
