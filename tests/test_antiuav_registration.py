import copy
import hashlib
import json
from pathlib import Path

import pytest

from aero_ir.data.antiuav_registration import (
    build_registration_audit,
    verify_registration_audit,
)


def _box(cx, cy, width, height, image_width, image_height):
    return [
        (cx - width / 2) * image_width,
        (cy - height / 2) * image_height,
        width * image_width,
        height * image_height,
    ]


def _write_release(root: Path, *, validation_offset: float = 0.0) -> None:
    label_root = root / "label_new"
    label_root.mkdir(parents=True)
    points = [
        (0.2, 0.3, 0.04, 0.05),
        (0.4, 0.2, 0.05, 0.04),
        (0.6, 0.5, 0.03, 0.06),
        (0.8, 0.7, 0.06, 0.03),
        (0.3, 0.8, 0.04, 0.04),
        (0.7, 0.35, 0.05, 0.05),
    ]
    for split in ("train", "val", "test"):
        sequences = [f"{split}-a", f"{split}-b"]
        (label_root / f"{split}.json").write_text(
            json.dumps({sequence: [] for sequence in sequences}),
            encoding="utf-8",
        )
        offset = validation_offset if split != "train" else 0.0
        for sequence in sequences:
            sequence_root = root / split / sequence
            sequence_root.mkdir(parents=True)
            visible_boxes = [
                _box(cx, cy, width, height, 1920, 1080) for cx, cy, width, height in points
            ]
            infrared_boxes = [
                _box(cx + offset, cy, width, height, 640, 512) for cx, cy, width, height in points
            ]
            labels = [1] * len(points)
            (sequence_root / "visible.json").write_text(
                json.dumps({"exist": labels, "gt_rect": visible_boxes}),
                encoding="utf-8",
            )
            (sequence_root / "infrared.json").write_text(
                json.dumps({"exist": labels, "gt_rect": infrared_boxes}),
                encoding="utf-8",
            )


def test_train_only_registration_can_qualify_target_boxes_but_not_dense_images(tmp_path):
    root = tmp_path / "Anti-UAV300"
    _write_release(root)

    report = build_registration_audit(root)

    assert report["data_usage"]["fit"] == "train only"
    assert report["metrics"]["val"]["frame_pass_rate"]["joint"] == pytest.approx(1.0)
    assert report["gates"]["train_fit_sanity"] == "pass"
    assert report["gates"]["held_out_validation"] == "pass"
    assert report["gates"]["calibrated_target_box_transfer"] == "pass"
    assert report["gates"]["dense_paired_image_registration"] == "hold"
    assert report["gates"]["generator_training_eligible"] == "hold"
    verify_registration_audit(report)


def test_held_out_geometry_shift_keeps_transfer_on_hold(tmp_path):
    root = tmp_path / "Anti-UAV300"
    _write_release(root, validation_offset=0.2)

    report = build_registration_audit(root)

    assert report["gates"]["train_fit_sanity"] == "pass"
    assert report["gates"]["held_out_validation"] == "hold"
    assert report["gates"]["calibrated_target_box_transfer"] == "hold"


def test_registration_audit_hash_rejects_tampering(tmp_path):
    root = tmp_path / "Anti-UAV300"
    _write_release(root)
    report = build_registration_audit(root)
    tampered = copy.deepcopy(report)
    tampered["model"]["fit_split"] = "test"

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_registration_audit(tampered)

    unsigned = {key: value for key, value in tampered.items() if key != "registration_audit_sha256"}
    payload = json.dumps(
        unsigned, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    tampered["registration_audit_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="fit on train"):
        verify_registration_audit(tampered)
