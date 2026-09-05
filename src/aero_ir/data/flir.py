"""Teledyne FLIR ADAS v2 metadata loading and integrity checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

FLIR_SPLITS = {
    "train": "images_thermal_train",
    "val": "images_thermal_val",
    "test": "video_thermal_test",
}
FLIR_RGB_SPLITS = {
    "train": "images_rgb_train",
    "val": "images_rgb_val",
    "test": "video_rgb_test",
}

FLIR_MANIFEST_SCHEMA = 1

_FRAME_INDEX = re.compile(r"-frame-(\d+)-")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_coco(root: Path, split_dir: str) -> dict:
    """Load one official COCO annotation file."""
    path = Path(root) / split_dir / "coco.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_detector_manifest(manifest: dict) -> None:
    """Reject a changed or malformed FLIR detector manifest."""
    if manifest.get("schema_version") != FLIR_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported FLIR manifest schema: {manifest.get('schema_version')}")
    recorded = manifest.get("manifest_sha256")
    if not isinstance(recorded, str):
        raise ValueError("FLIR manifest has no manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = _canonical_sha256(unsigned)
    if actual != recorded:
        raise ValueError(f"FLIR manifest hash mismatch: expected {recorded}, got {actual}")


def build_detector_manifest(
    root: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    representation: str = "analytics16",
    classes: tuple[str, ...] = ("person", "bike", "car", "motor", "bus", "truck"),
    audit_report: dict | None = None,
) -> dict:
    """Build the deterministic, path-relative manifest used by detector experiments.

    All source image and annotation identifiers are retained. Detector labels are also mapped
    to contiguous zero-based indices without changing the original COCO category ids. Raw files
    are anchored by the audited source-archive and annotation hashes rather than by mutable
    absolute paths.
    """
    root = Path(root)
    if representation not in {"analytics16", "display8"}:
        raise ValueError("representation must be analytics16 or display8")
    if not splits or any(split not in FLIR_SPLITS for split in splits):
        raise ValueError(f"splits must be drawn from {sorted(FLIR_SPLITS)}")
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("classes must be non-empty and unique")
    if audit_report is not None:
        audit_root = Path(audit_report.get("root", ""))
        if audit_root.resolve() != root.resolve():
            raise ValueError(f"audit root {audit_root} does not match dataset root {root}")
        gate = audit_report.get("gates", {}).get("thermal_baseline_data_integrity")
        if gate != "pass":
            raise ValueError(f"thermal_baseline_data_integrity must pass, got {gate!r}")

    manifest_splits: dict[str, dict] = {}
    category_spec: list[dict] | None = None
    for split in splits:
        split_dir = FLIR_SPLITS[split]
        split_root = root / split_dir
        coco_path = split_root / "coco.json"
        coco = load_coco(root, split_dir)
        category_by_name = {
            str(category["name"]): int(category["id"]) for category in coco.get("categories", [])
        }
        missing_classes = [name for name in classes if name not in category_by_name]
        if missing_classes:
            raise ValueError(f"classes absent from {split} COCO categories: {missing_classes}")
        selected_categories = [
            {
                "name": name,
                "source_category_id": category_by_name[name],
                "class_index": index,
            }
            for index, name in enumerate(classes)
        ]
        if category_spec is None:
            category_spec = selected_categories
        elif category_spec != selected_categories:
            raise ValueError(f"category mapping differs in split {split}")

        class_index_by_source_id = {
            category["source_category_id"]: category["class_index"]
            for category in selected_categories
        }
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        selected_annotation_count = 0
        excluded_annotation_count = 0
        for annotation in sorted(coco.get("annotations", []), key=lambda item: int(item["id"])):
            source_category_id = int(annotation["category_id"])
            if source_category_id not in class_index_by_source_id:
                excluded_annotation_count += 1
                continue
            bbox = [float(value) for value in annotation["bbox"]]
            record = {
                "annotation_id": int(annotation["id"]),
                "source_category_id": source_category_id,
                "class_index": class_index_by_source_id[source_category_id],
                "bbox_xywh": bbox,
                "area": float(annotation.get("area", bbox[2] * bbox[3])),
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
            annotations_by_image[int(annotation["image_id"])].append(record)
            selected_annotation_count += 1

        records: list[dict] = []
        annotated_images = 0
        for image in sorted(coco.get("images", []), key=lambda item: int(item["id"])):
            source_relative = Path(image["file_name"])
            if representation == "analytics16":
                image_relative = (
                    Path(split_dir) / "analyticsData" / source_relative.with_suffix(".tiff").name
                )
            else:
                image_relative = Path(split_dir) / source_relative
            if not (root / image_relative).is_file():
                raise FileNotFoundError(root / image_relative)
            image_id = int(image["id"])
            annotations = annotations_by_image.get(image_id, [])
            annotated_images += bool(annotations)
            records.append(
                {
                    "image_id": image_id,
                    "image_path": image_relative.as_posix(),
                    "source_file_name": source_relative.as_posix(),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "video_id": image.get("extra_info", {}).get("video_id"),
                    "annotations": annotations,
                }
            )

        manifest_splits[split] = {
            "source_annotation_path": f"{split_dir}/coco.json",
            "source_annotation_sha256": sha256_file(coco_path),
            "counts": {
                "images": len(records),
                "images_with_selected_annotations": annotated_images,
                "selected_annotations": selected_annotation_count,
                "excluded_annotations": excluded_annotation_count,
            },
            "records": records,
        }

    manifest = {
        "schema_version": FLIR_MANIFEST_SCHEMA,
        "dataset": "Teledyne FLIR ADAS Thermal Dataset v2",
        "representation": representation,
        "categories": category_spec,
        "source_archive": audit_report.get("source_archive") if audit_report else None,
        "splits": manifest_splits,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    verify_detector_manifest(manifest)
    return manifest


def _frame_index(filename: str) -> int | None:
    match = _FRAME_INDEX.search(filename)
    return int(match.group(1)) if match else None


def audit_coco_split(root: Path, split_dir: str) -> tuple[dict, set[str]]:
    """Check file, id, category and bounding-box integrity for one split."""
    root = Path(root)
    split_root = root / split_dir
    coco_path = split_root / "coco.json"
    coco = load_coco(root, split_dir)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_ids = [image.get("id") for image in images]
    image_by_id = {image.get("id"): image for image in images}
    category_ids = {category.get("id") for category in categories}
    filenames = [image.get("file_name", "") for image in images]
    missing_files = [name for name in filenames if not (split_root / name).is_file()]
    listed_basenames = {Path(name).name for name in filenames}
    disk_basenames = {path.name for path in (split_root / "data").glob("*.jpg")}

    bad_image_refs = 0
    bad_category_refs = 0
    invalid_boxes = 0
    out_of_bounds_boxes = 0
    annotated_image_ids: set[int] = set()
    for annotation in annotations:
        image_id = annotation.get("image_id")
        image = image_by_id.get(image_id)
        if image is None:
            bad_image_refs += 1
            continue
        annotated_image_ids.add(image_id)
        if annotation.get("category_id") not in category_ids:
            bad_category_refs += 1
        box = annotation.get("bbox", [])
        if len(box) != 4 or not np.isfinite(box).all():
            invalid_boxes += 1
            continue
        x, y, width, height = (float(value) for value in box)
        if width <= 0 or height <= 0:
            invalid_boxes += 1
            continue
        if (
            x < 0
            or y < 0
            or x + width > float(image["width"]) + 1e-6
            or y + height > float(image["height"]) + 1e-6
        ):
            out_of_bounds_boxes += 1

    analytics_root = split_root / "analyticsData"
    analytics_basenames = (
        {path.stem for path in analytics_root.glob("*.tiff")} if analytics_root.is_dir() else set()
    )
    missing_analytics = (
        sorted(Path(name).stem for name in filenames if Path(name).stem not in analytics_basenames)
        if analytics_root.is_dir()
        else []
    )
    video_ids = {
        str(image.get("extra_info", {}).get("video_id"))
        for image in images
        if image.get("extra_info", {}).get("video_id") is not None
    }
    report = {
        "annotation_sha256": sha256_file(coco_path),
        "images_in_coco": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "video_ids": len(video_ids),
        "duplicate_image_ids": len(image_ids) - len(set(image_ids)),
        "duplicate_filenames": len(filenames) - len(set(filenames)),
        "missing_image_files": len(missing_files),
        "orphan_jpegs": len(disk_basenames - listed_basenames),
        "images_without_annotations": len(set(image_ids) - annotated_image_ids),
        "annotations_with_missing_image": bad_image_refs,
        "annotations_with_unknown_category": bad_category_refs,
        "invalid_boxes": invalid_boxes,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "analytics_tiffs": len(analytics_basenames),
        "missing_analytics_tiffs": len(missing_analytics),
    }
    return report, video_ids


def audit_video_pairs(root: Path, map_path: Path) -> dict:
    """Validate the official RGB-to-thermal test-video mapping."""
    root = Path(root)
    map_path = Path(map_path)
    with map_path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    rgb_root = root / "video_rgb_test" / "data"
    thermal_root = root / "video_thermal_test" / "data"
    missing_rgb = [name for name in mapping if not (rgb_root / name).is_file()]
    missing_thermal = [name for name in mapping.values() if not (thermal_root / name).is_file()]
    frame_mismatches = [
        [rgb_name, thermal_name]
        for rgb_name, thermal_name in mapping.items()
        if _frame_index(rgb_name) != _frame_index(thermal_name)
    ]
    return {
        "mapping_sha256": sha256_file(map_path),
        "pairs": len(mapping),
        "unique_thermal_targets": len(set(mapping.values())),
        "duplicate_thermal_targets": len(mapping) - len(set(mapping.values())),
        "missing_rgb_files": len(missing_rgb),
        "missing_thermal_files": len(missing_thermal),
        "frame_index_mismatches": len(frame_mismatches),
        "geometric_registration_audited": False,
    }


def audit_release(root: Path, video_map: Path, archive: Path | None = None) -> dict:
    """Audit the official thermal splits and paired video manifest."""
    split_reports: dict[str, dict] = {}
    video_ids: dict[str, set[str]] = {}
    for split_name, split_dir in FLIR_SPLITS.items():
        split_reports[split_name], video_ids[split_name] = audit_coco_split(root, split_dir)

    rgb_split_reports: dict[str, dict] = {}
    rgb_video_ids: dict[str, set[str]] = {}
    for split_name, split_dir in FLIR_RGB_SPLITS.items():
        rgb_split_reports[split_name], rgb_video_ids[split_name] = audit_coco_split(root, split_dir)

    overlap = {
        "train_val": sorted(video_ids["train"] & video_ids["val"]),
        "train_test": sorted(video_ids["train"] & video_ids["test"]),
        "val_test": sorted(video_ids["val"] & video_ids["test"]),
    }
    pair_report = audit_video_pairs(root, video_map)
    hard_integrity_failures = sum(
        report[key]
        for report in split_reports.values()
        for key in (
            "duplicate_image_ids",
            "duplicate_filenames",
            "missing_image_files",
            "annotations_with_missing_image",
            "annotations_with_unknown_category",
            "invalid_boxes",
            "missing_analytics_tiffs",
        )
    )
    pair_failures = sum(
        pair_report[key]
        for key in (
            "missing_rgb_files",
            "missing_thermal_files",
            "frame_index_mismatches",
            "duplicate_thermal_targets",
        )
    )
    sequence_overlap = sum(len(values) for values in overlap.values())
    rgb_sequence_overlap = {
        "train_val": sorted(rgb_video_ids["train"] & rgb_video_ids["val"]),
        "train_test": sorted(rgb_video_ids["train"] & rgb_video_ids["test"]),
        "val_test": sorted(rgb_video_ids["val"] & rgb_video_ids["test"]),
    }
    rgb_hard_integrity_failures = sum(
        report[key]
        for report in rgb_split_reports.values()
        for key in (
            "duplicate_image_ids",
            "duplicate_filenames",
            "missing_image_files",
            "annotations_with_missing_image",
            "annotations_with_unknown_category",
            "invalid_boxes",
        )
    )
    rgb_overlap_count = sum(len(values) for values in rgb_sequence_overlap.values())
    archive_report = None
    if archive is not None:
        archive = Path(archive)
        archive_report = {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    return {
        "root": str(Path(root).resolve()),
        "source_archive": archive_report,
        "thermal_splits": split_reports,
        "rgb_splits": rgb_split_reports,
        "thermal_sequence_overlap": overlap,
        "rgb_sequence_overlap": rgb_sequence_overlap,
        "official_video_pairs": pair_report,
        "gates": {
            "thermal_baseline_data_integrity": (
                "pass" if hard_integrity_failures == 0 and sequence_overlap == 0 else "hold"
            ),
            "rgb_source_data_integrity": (
                "pass" if rgb_hard_integrity_failures == 0 and rgb_overlap_count == 0 else "hold"
            ),
            "time_synchronised_video_pairs": "pass" if pair_failures == 0 else "hold",
            "label_transfer_registration": "hold",
        },
    }


def load_thermal_sample(
    root: Path,
    split: str,
    *,
    n_images: int,
    seed: int,
    representation: str = "analytics16",
    exclude_image_ids: set[int] | None = None,
) -> tuple[list[np.ndarray], list[list[list[float]]], list[int]]:
    """Load a deterministic annotated thermal sample for a data pilot."""
    if split not in FLIR_SPLITS:
        raise ValueError(f"unknown split: {split}")
    if representation not in {"analytics16", "display8"}:
        raise ValueError("representation must be analytics16 or display8")
    if n_images <= 0:
        raise ValueError("n_images must be positive")

    split_dir = FLIR_SPLITS[split]
    split_root = Path(root) / split_dir
    coco = load_coco(root, split_dir)
    boxes_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        box = annotation.get("bbox", [])
        if len(box) == 4 and float(box[2]) > 0 and float(box[3]) > 0:
            boxes_by_image[int(annotation["image_id"])].append([float(value) for value in box])

    excluded = exclude_image_ids or set()
    candidates = [
        image
        for image in coco.get("images", [])
        if int(image["id"]) in boxes_by_image and int(image["id"]) not in excluded
    ]
    if n_images > len(candidates):
        raise ValueError(f"requested {n_images} images from only {len(candidates)} candidates")
    rng = np.random.default_rng(seed)
    selected = [candidates[index] for index in sorted(rng.choice(len(candidates), n_images, False))]

    images: list[np.ndarray] = []
    boxes: list[list[list[float]]] = []
    image_ids: list[int] = []
    for image in selected:
        filename = Path(image["file_name"])
        path = split_root / filename
        if representation == "analytics16":
            path = split_root / "analyticsData" / filename.with_suffix(".tiff").name
        with Image.open(path) as source:
            array = np.asarray(source, dtype=np.float64)
        if array.shape != (int(image["height"]), int(image["width"])):
            raise ValueError(f"shape mismatch for {path}: {array.shape}")
        image_id = int(image["id"])
        images.append(array)
        boxes.append(boxes_by_image[image_id])
        image_ids.append(image_id)
    return images, boxes, image_ids
