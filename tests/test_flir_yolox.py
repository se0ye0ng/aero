import json

import numpy as np
import pytest
from test_flir import _write_release

from aero_ir.data.flir import audit_release, build_detector_manifest
from aero_ir.data.preprocess import fit_uint16_linear_preprocess
from aero_ir.data.registry import FLIRThermalDataset
from aero_ir.detect.flir_yolox import (
    build_flir_yolox_dataset,
    export_coco_subset,
    export_yolox_coco,
)


def test_manifest_locked_yolox_view_loads_analytics16(tmp_path):
    pytest.importorskip("yolox")
    root = tmp_path / "FLIR_ADAS_v2"
    video_map = _write_release(root)
    manifest = build_detector_manifest(
        root,
        classes=("person",),
        audit_report=audit_release(root, video_map),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry_dataset = FLIRThermalDataset(root, manifest_path, "train")
    preprocessing = fit_uint16_linear_preprocess(
        registry_dataset,
        manifest_sha256=manifest["manifest_sha256"],
        sample_size=1,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )
    preprocess_path = tmp_path / "preprocess.json"
    preprocess_path.write_text(json.dumps(preprocessing), encoding="utf-8")

    prepared_root = tmp_path / "prepared"
    report = export_yolox_coco(manifest_path, prepared_root)
    dataset = build_flir_yolox_dataset(
        image_root=root,
        prepared_root=prepared_root,
        annotation_file="train.json",
        preprocess_path=preprocess_path,
        image_size=(16, 16),
    )

    image = dataset.load_image(0)
    assert report["train"]["images"] == 1
    assert len(dataset) == 1
    assert image.dtype == np.uint8
    assert image.shape == (8, 10, 3)
    assert np.array_equal(image[:, :, 0], image[:, :, 2])
    assert dataset.load_anno(0).tolist() == [[1.6, 1.6, 6.4, 8.0, 0.0]]

    subset_path = prepared_root / "annotations" / "smoke_train.json"
    subset = export_coco_subset(
        prepared_root / "annotations" / "train.json",
        subset_path,
        n_images=1,
        seed=0,
    )
    assert subset["images"] == 1
    assert json.loads(subset_path.read_text())["info"]["engineering_smoke_only"]
