"""Manifest-aware FLIR input adapter for the pinned upstream YOLOX package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from aero_ir.data.flir import verify_detector_manifest
from aero_ir.data.preprocess import (
    apply_uint16_linear_preprocess,
    verify_preprocess_spec,
)
from aero_ir.utils.manifest import file_sha256


def export_yolox_coco(manifest_path: Path, output_root: Path) -> dict[str, dict[str, str | int]]:
    """Export filtered COCO annotations without copying or modifying source imagery."""
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_detector_manifest(manifest)
    categories = [
        {"id": item["source_category_id"], "name": item["name"]} for item in manifest["categories"]
    ]
    annotation_root = output_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, str | int]] = {}
    for split, split_manifest in manifest["splits"].items():
        images: list[dict] = []
        annotations: list[dict] = []
        for record in split_manifest["records"]:
            images.append(
                {
                    "id": record["image_id"],
                    "file_name": record["image_path"],
                    "width": record["width"],
                    "height": record["height"],
                }
            )
            for annotation in record["annotations"]:
                annotations.append(
                    {
                        "id": annotation["annotation_id"],
                        "image_id": record["image_id"],
                        "category_id": annotation["source_category_id"],
                        "bbox": annotation["bbox_xywh"],
                        "area": annotation["area"],
                        "iscrowd": annotation["iscrowd"],
                    }
                )
        payload = {
            "info": {
                "aero_manifest_sha256": manifest["manifest_sha256"],
                "representation": manifest["representation"],
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        destination = annotation_root / f"{split}.json"
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report[split] = {
            "path": str(destination),
            "sha256": file_sha256(destination),
            "images": len(images),
            "annotations": len(annotations),
        }
    return report


def export_coco_subset(
    source: Path,
    destination: Path,
    *,
    n_images: int,
    seed: int,
) -> dict[str, str | int]:
    """Write a deterministic engineering-only subset of a prepared COCO split."""
    source = Path(source)
    destination = Path(destination)
    payload = json.loads(source.read_text(encoding="utf-8"))
    images = payload["images"]
    if n_images <= 0 or n_images > len(images):
        raise ValueError(f"n_images must be in [1, {len(images)}]")
    rng = np.random.default_rng(seed)
    indices = sorted(int(index) for index in rng.choice(len(images), n_images, replace=False))
    selected_images = [images[index] for index in indices]
    selected_ids = {image["id"] for image in selected_images}
    selected_annotations = [
        annotation
        for annotation in payload["annotations"]
        if annotation["image_id"] in selected_ids
    ]
    subset = {
        **payload,
        "info": {
            **payload.get("info", {}),
            "engineering_smoke_only": True,
            "selection_seed": seed,
            "selection_source_sha256": file_sha256(source),
        },
        "images": selected_images,
        "annotations": selected_annotations,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(destination),
        "sha256": file_sha256(destination),
        "images": len(selected_images),
        "annotations": len(selected_annotations),
    }


def build_flir_yolox_dataset(
    *,
    image_root: Path,
    prepared_root: Path,
    annotation_file: str,
    preprocess_path: Path,
    image_size: tuple[int, int] = (640, 640),
    transform=None,
):
    """Create an upstream dataset whose source read preserves analytics16 semantics."""
    try:
        from yolox.data import COCODataset
    except ImportError as error:  # pragma: no cover - minimal environments
        raise RuntimeError("FLIR YOLOX loading requires the pinned YOLOX backend") from error

    specification = json.loads(Path(preprocess_path).read_text(encoding="utf-8"))
    annotation_path = Path(prepared_root) / "annotations" / annotation_file
    annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    manifest_sha256 = annotation_payload.get("info", {}).get("aero_manifest_sha256")
    verify_preprocess_spec(specification, expected_manifest_sha256=manifest_sha256)

    class ManifestLockedCOCODataset(COCODataset):
        def load_image(self, index):
            file_name = self.annotations[index][3]
            image_path = Path(image_root) / file_name
            with Image.open(image_path) as source:
                image = np.asarray(source).copy()
            mapped = apply_uint16_linear_preprocess(image, specification)
            return np.rint(mapped).astype(np.uint8)

    return ManifestLockedCOCODataset(
        data_dir=str(prepared_root),
        json_file=annotation_file,
        name="",
        img_size=image_size,
        preproc=transform,
        cache=False,
    )
