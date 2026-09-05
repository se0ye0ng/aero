"""Dataset registry.

Datasets are declared in ``configs/data/*.yaml`` and resolved here. Nothing is downloaded
automatically from a gated source; ``scripts/download_*.sh`` prints the access route and
verifies checksums after manual placement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from aero_ir.data.flir import verify_detector_manifest

DATASETS = {
    "flir_urban": "Teledyne FLIR ADAS Thermal v2 - IR protocol-transfer domain",
    "llvip": "LLVIP - aligned low-light visible-infrared pairs",
    "antiuav_small": "Anti-UAV300 paired source + Anti-UAV410 external IR evaluation",
    "dronevehicle": "DroneVehicle - aerial RGB-IR, oriented boxes",
}


def _cfg_value(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class FLIRThermalDataset:
    """Lazy, manifest-locked FLIR thermal detection dataset.

    Images remain single-channel and retain their source bit depth. Detector-specific resizing,
    channel replication and intensity scaling belong in the frozen model preprocessing step.
    """

    def __init__(self, root: Path, manifest_path: Path, split: str) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        verify_detector_manifest(self.manifest)
        if split not in self.manifest["splits"]:
            raise ValueError(f"split {split!r} is absent from {self.manifest_path}")
        self.split = split
        self.representation = self.manifest["representation"]
        self.categories = tuple(self.manifest["categories"])
        self.records = tuple(self.manifest["splits"][split]["records"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict]:
        record = self.records[index]
        path = self.root / record["image_path"]
        with Image.open(path) as source:
            image = np.asarray(source).copy()
        expected_shape = (record["height"], record["width"])
        if image.shape != expected_shape:
            raise ValueError(
                f"shape mismatch for {path}: expected {expected_shape}, got {image.shape}"
            )
        annotations = record["annotations"]
        target = {
            "image_id": int(record["image_id"]),
            "boxes_xywh": np.asarray(
                [annotation["bbox_xywh"] for annotation in annotations], dtype=np.float32
            ).reshape(-1, 4),
            "labels": np.asarray(
                [annotation["class_index"] for annotation in annotations], dtype=np.int64
            ),
            "source_category_ids": np.asarray(
                [annotation["source_category_id"] for annotation in annotations], dtype=np.int64
            ),
            "video_id": record["video_id"],
            "image_path": str(path),
            "representation": self.representation,
        }
        return image, target


def build_dataset(cfg, split: str = "train") -> FLIRThermalDataset:
    """Build a lazy dataset from an audited, content-addressed manifest."""
    name = _cfg_value(cfg, "name")
    if name != "flir_urban":
        if name in DATASETS:
            raise NotImplementedError(f"dataset adapter is not implemented for {name!r}")
        raise ValueError(f"unknown dataset: {name!r}")
    root = _cfg_value(cfg, "root")
    manifest_path = _cfg_value(cfg, "manifest")
    if not root:
        raise ValueError("FLIR dataset config requires root")
    if not manifest_path:
        raise ValueError("FLIR dataset config requires an immutable manifest")
    return FLIRThermalDataset(Path(root), Path(manifest_path), split)
