"""YOLOX dataset adapter for prepared Anti-UAV300 native-IR PNG frames."""

from __future__ import annotations

import json
from pathlib import Path

from aero_ir.data.antiuav300_ir import verify_native_ir_manifest


def build_antiuav300_yolox_dataset(
    *,
    prepared_root: Path,
    annotation_file: str,
    image_size: tuple[int, int] = (640, 640),
    transform=None,
):
    """Create an upstream COCO dataset after checking its frozen subset identity."""
    try:
        from yolox.data import COCODataset
    except ImportError as error:  # pragma: no cover - minimal environments
        raise RuntimeError("Anti-UAV300 YOLOX loading requires the pinned backend") from error

    prepared_root = Path(prepared_root).resolve()
    manifest_path = prepared_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_native_ir_manifest(manifest)
    annotation_path = prepared_root / "annotations" / annotation_file
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    info = annotation.get("info", {})
    if info.get("aero_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Anti-UAV300 COCO annotation is not bound to the prepared manifest")
    split = info.get("split")
    if split not in {"train", "val"} or annotation_file != f"{split}.json":
        raise ValueError("Anti-UAV300 COCO annotation filename/split mismatch")

    return COCODataset(
        data_dir=str(prepared_root),
        json_file=annotation_file,
        name="",
        img_size=image_size,
        preproc=transform,
        cache=False,
    )
