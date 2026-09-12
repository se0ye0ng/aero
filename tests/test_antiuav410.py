import hashlib
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from aero_ir.data.antiuav410 import (
    ANTIUAV410_ATTRIBUTE_KEYS,
    audit_release,
    build_external_manifest,
    export_external_coco,
    target_pixel_area_bin,
    verify_external_manifest,
)
from aero_ir.data.registry import AntiUAV410Dataset, build_dataset


def _write_release(root, archive_path):
    for split, sequence in (
        ("train", "train-sequence"),
        ("val", "val-sequence"),
        ("test", "test-sequence"),
    ):
        sequence_root = root / split / sequence
        sequence_root.mkdir(parents=True)
        for frame_index in range(1, 5):
            image = np.full((512, 640), frame_index, dtype=np.uint8)
            Image.fromarray(np.repeat(image[..., None], 3, axis=2)).save(
                sequence_root / f"{frame_index}.jpg"
            )
        labels = {
            "exist": [1, 1, 0, 0],
            "gt_rect": [
                [10, 20, 4, 5],
                [10, 20, 0, 5],
                [1, 1, 2, 2],
                [0, 0, 0, 0],
            ],
            ANTIUAV410_ATTRIBUTE_KEYS[0]: [0, 1, 0, 0],
        }
        (sequence_root / "IR_label.json").write_text(json.dumps(labels), encoding="utf-8")

    with zipfile.ZipFile(archive_path, "w") as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())


def test_release_audit_qualifies_filtered_external_evaluation(tmp_path):
    root = tmp_path / "Anti-UAV410"
    archive = tmp_path / "Anti-UAV410.zip"
    _write_release(root, archive)

    report = audit_release(root, archive=archive)

    assert report["gates"]["archive_integrity"] == "pass"
    assert report["gates"]["extraction_completeness"] == "pass"
    assert report["gates"]["image_storage_contract"] == "pass"
    assert report["gates"]["official_split_sequence_disjointness"] == "pass"
    assert report["gates"]["annotation_integrity"] == "hold"
    assert report["gates"]["attribute_completeness"] == "hold"
    assert report["gates"]["external_evaluation_eligible"] == "pass"
    assert report["gates"]["generator_or_curation_eligible"] == "hold"
    assert report["splits"]["test"]["images"] == 4
    assert report["splits"]["test"]["invalid_present_boxes"] == 1
    assert report["splits"]["test"]["absent_frames_with_nonzero_box"] == 1


def test_manifest_excludes_invalid_positives_and_keeps_negatives(tmp_path):
    root = tmp_path / "Anti-UAV410"
    archive = tmp_path / "Anti-UAV410.zip"
    _write_release(root, archive)
    audit = audit_release(root, archive=archive)

    first = build_external_manifest(root, audit_report=audit)
    second = build_external_manifest(root, audit_report=audit)

    assert first == second
    verify_external_manifest(first)
    split = first["splits"]["test"]
    assert split["counts"] == {
        "absent_frames_with_nonzero_source_box": 1,
        "annotations": 1,
        "excluded_invalid_present_images": 1,
        "included_images": 3,
        "negative_images": 2,
        "positive_images": 1,
        "source_frames": 4,
    }
    assert split["exclusion_counts"] == {"non_positive_extent": 1}
    assert [record["frame_index"] for record in split["records"]] == [1, 3, 4]
    assert [record["visibility"] for record in split["records"]] == [
        "present",
        "absent",
        "absent",
    ]
    assert split["records"][0]["target_pixel_area"] == 20.0
    assert split["records"][0]["target_pixel_area_bin"] == "small"


def test_coco_export_and_lazy_registry_are_external_only(tmp_path):
    root = tmp_path / "Anti-UAV410"
    archive = tmp_path / "Anti-UAV410.zip"
    _write_release(root, archive)
    manifest = build_external_manifest(root, audit_report=audit_release(root, archive=archive))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    coco_path = tmp_path / "coco" / "test.json"

    export = export_external_coco(manifest_path, coco_path)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    assert export["images"] == 3
    assert export["annotations"] == 1
    assert coco["images"][1]["visibility"] == "absent"
    assert coco["images"][1]["video_id"] == "test-sequence"

    dataset = AntiUAV410Dataset(root, manifest_path)
    image, target = dataset[0]
    assert image.shape == (512, 640, 3)
    assert image.dtype == np.uint8
    assert target["boxes_xywh"].tolist() == [[10.0, 20.0, 4.0, 5.0]]
    assert target["sequence_id"] == "test-sequence"
    assert target["target_pixel_area_bin"] == "small"
    assert target["external_evaluation_only"]

    cfg = {
        "name": "antiuav_small",
        "external_eval_root": str(root),
        "external_eval_manifest": str(manifest_path),
    }
    assert len(build_dataset(cfg, split="external_test")) == 3
    with pytest.raises(NotImplementedError, match="registration-gated"):
        build_dataset(cfg, split="train")


def test_manifest_rejects_tampering_and_area_bins_are_frozen(tmp_path):
    root = tmp_path / "Anti-UAV410"
    archive = tmp_path / "Anti-UAV410.zip"
    _write_release(root, archive)
    manifest = build_external_manifest(root, audit_report=audit_release(root, archive=archive))
    manifest["splits"]["test"]["records"][0]["frame_index"] = 99

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_external_manifest(manifest)
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    payload = json.dumps(
        unsigned, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="frame index/path mismatch"):
        verify_external_manifest(manifest)
    assert target_pixel_area_bin(15) == "tiny"
    assert target_pixel_area_bin(16) == "small"
    assert target_pixel_area_bin(1023) == "small"
    assert target_pixel_area_bin(1024) == "medium"
    assert target_pixel_area_bin(9216) == "large"
    with pytest.raises(ValueError, match="positive"):
        target_pixel_area_bin(0)
