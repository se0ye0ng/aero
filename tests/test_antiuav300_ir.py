import json
from pathlib import Path

import numpy as np
import pytest

from aero_ir.data import antiuav300_ir
from aero_ir.data.antiuav300_ir import (
    export_native_ir_coco,
    prepare_native_ir_smoke,
    uniform_frame_indices,
    verify_native_ir_manifest,
)
from aero_ir.detect.antiuav300_yolox import build_antiuav300_yolox_dataset
from aero_ir.utils.manifest import file_sha256


def _write_source(root: Path) -> None:
    labels_root = root / "label_new"
    labels_root.mkdir(parents=True)
    for split in ("train", "val"):
        sequence = f"{split}-sequence"
        (labels_root / f"{split}.json").write_text(
            json.dumps({sequence: ["small", "day"]}), encoding="utf-8"
        )
        sequence_root = root / split / sequence
        sequence_root.mkdir(parents=True)
        labels = {
            "exist": [1, 0, 1, 1, 0],
            "gt_rect": [
                [10, 20, 4, 5],
                [0, 0, 0, 0],
                [10, 20, 0, 5],
                [12, 22, 4, 5],
                [1, 1, 2, 2],
            ],
        }
        (sequence_root / "infrared.json").write_text(json.dumps(labels), encoding="utf-8")
        (sequence_root / "infrared.mp4").touch()


def _audit(root: Path) -> dict:
    return {
        "root": str(root.resolve()),
        "source_archive": {"sha256": "a" * 64},
        "gates": {
            "archive_integrity": "pass",
            "extraction_completeness": "pass",
            "official_manifest_sequence_disjointness": "pass",
            "paired_temporal_integrity": "pass",
            "train_only_pilot_eligible": "pass",
            "annotation_integrity": "hold",
        },
    }


def _fake_decode(video_path, selected_indices, destinations, *, width, height):
    assert video_path.name == "infrared.mp4"
    assert (width, height) == (640, 512)
    result = {}
    for frame_index in selected_indices:
        destination = destinations[frame_index]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(np.asarray([frame_index], dtype=np.uint8).tobytes())
        result[frame_index] = {
            "prepared_image_sha256": file_sha256(destination),
            "prepared_image_bytes": destination.stat().st_size,
        }
    return result


def test_uniform_selection_is_label_independent_and_endpoint_inclusive():
    assert uniform_frame_indices(10, 4) == [0, 3, 6, 9]
    assert uniform_frame_indices(7, 1) == [3]
    with pytest.raises(ValueError, match="sample_count"):
        uniform_frame_indices(2, 3)


def test_native_ir_manifest_filters_invalid_selected_positive_and_keeps_negative(
    tmp_path, monkeypatch
):
    root = tmp_path / "Anti-UAV300"
    output = tmp_path / "prepared"
    _write_source(root)
    monkeypatch.setattr(
        antiuav300_ir,
        "probe_video",
        lambda path: {"opened": True, "frames": 5, "width": 640, "height": 512, "fps": 20.0},
    )
    monkeypatch.setattr(antiuav300_ir, "_decode_selected_frames", _fake_decode)

    first = prepare_native_ir_smoke(
        root,
        output,
        audit_report=_audit(root),
        samples_per_sequence={"train": 3, "val": 2},
    )
    second = prepare_native_ir_smoke(
        root,
        output,
        audit_report=_audit(root),
        samples_per_sequence={"train": 3, "val": 2},
    )

    assert first == second
    verify_native_ir_manifest(first, prepared_root=output)
    train = first["splits"]["train"]
    assert train["counts"]["selected_source_frames"] == 3
    assert train["counts"]["included_images"] == 2
    assert train["counts"]["positive_images"] == 1
    assert train["counts"]["negative_images"] == 1
    assert train["counts"]["absent_frames_with_nonzero_source_box"] == 1
    assert train["counts"]["excluded_invalid_present_images"] == 1
    assert train["counts"]["excluded_non_positive_extent"] == 1
    assert train["exclusions"][0]["frame_index_zero_based"] == 2
    assert first["split_usage"]["test"] == "not_accessed"

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(first), encoding="utf-8")
    report = export_native_ir_coco(manifest_path, output)
    coco = json.loads((output / "annotations" / "train.json").read_text(encoding="utf-8"))
    assert report["train"]["images"] == 2
    assert report["train"]["annotations"] == 1
    assert coco["images"][1]["visibility"] == "absent"
    assert coco["categories"] == [{"id": 1, "name": "uav", "supercategory": "aircraft"}]
    dataset = build_antiuav300_yolox_dataset(
        prepared_root=output,
        annotation_file="train.json",
    )
    assert len(dataset) == 2


def test_native_ir_manifest_rejects_gate_failure_and_tampering(tmp_path, monkeypatch):
    root = tmp_path / "Anti-UAV300"
    output = tmp_path / "prepared"
    _write_source(root)
    monkeypatch.setattr(
        antiuav300_ir,
        "probe_video",
        lambda path: {"opened": True, "frames": 5, "width": 640, "height": 512, "fps": 20.0},
    )
    monkeypatch.setattr(antiuav300_ir, "_decode_selected_frames", _fake_decode)
    bad_audit = _audit(root)
    bad_audit["gates"]["train_only_pilot_eligible"] = "hold"
    with pytest.raises(ValueError, match="does not qualify"):
        prepare_native_ir_smoke(
            root,
            output,
            audit_report=bad_audit,
            samples_per_sequence={"train": 3, "val": 2},
        )

    manifest = prepare_native_ir_smoke(
        root,
        output,
        audit_report=_audit(root),
        samples_per_sequence={"train": 3, "val": 2},
    )
    manifest["splits"]["train"]["records"][0]["frame_index_zero_based"] = 99
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_native_ir_manifest(manifest)
