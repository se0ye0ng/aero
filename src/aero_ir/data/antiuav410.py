"""Audited Anti-UAV410 external-evaluation manifests and lazy loading."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

from aero_ir.data.flir import sha256_file

ANTIUAV410_SPLITS = ("train", "val", "test")
ANTIUAV410_MANIFEST_SCHEMA = 1
ANTIUAV410_WIDTH = 640
ANTIUAV410_HEIGHT = 512
ANTIUAV410_CHANNELS = 3
ANTIUAV410_IMAGE_MODE = "RGB"
ANTIUAV410_ATTRIBUTE_KEYS = ("DBC", "FM", "IC", "LR", "OC", "SV", "TS", "VE")

# AERO analysis bins, fixed before external evaluation. The tiny bucket is below 16 px^2;
# the remaining boundaries follow COCO's conventional small/medium/large area thresholds.
TARGET_AREA_BIN_EDGES = (16, 32**2, 96**2)
TARGET_AREA_BIN_NAMES = ("tiny", "small", "medium", "large")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _image_names(sequence_root: Path) -> list[str]:
    names = [
        path.name
        for path in sequence_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    ]
    return sorted(
        names,
        key=lambda name: (
            (
                0,
                int(Path(name).stem),
            )
            if Path(name).stem.isdigit()
            else (1, name)
        ),
    )


def _box_status(
    box: object,
    *,
    width: int = ANTIUAV410_WIDTH,
    height: int = ANTIUAV410_HEIGHT,
) -> tuple[list[float] | None, str | None]:
    if not isinstance(box, list) or len(box) != 4:
        return None, "malformed"
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None, "malformed"
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite"
    x, y, box_width, box_height = values
    if box_width <= 0 or box_height <= 0:
        return None, "non_positive_extent"
    if x < 0 or y < 0 or x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
        return None, "out_of_bounds"
    return values, None


def _box_is_nonzero(box: object) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values) and any(value != 0 for value in values)


def target_pixel_area_bin(area: float) -> str:
    """Map a positive box area to the fixed AERO small-target analysis bins."""
    if not math.isfinite(area) or area <= 0:
        raise ValueError("target pixel area must be finite and positive")
    for edge, name in zip(TARGET_AREA_BIN_EDGES, TARGET_AREA_BIN_NAMES, strict=False):
        if area < edge:
            return name
    return TARGET_AREA_BIN_NAMES[-1]


def audit_split(root: Path, split: str) -> dict:
    """Audit one extracted Anti-UAV410 split without modifying source data."""
    if split not in ANTIUAV410_SPLITS:
        raise ValueError(f"unknown Anti-UAV410 split: {split}")
    split_root = Path(root) / split
    if not split_root.is_dir():
        return {
            "sequences": 0,
            "images": 0,
            "missing_split": True,
            "missing_label_files": [],
            "label_parse_errors": [],
        }

    totals: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    attribute_missing: Counter[str] = Counter()
    attribute_incomplete: Counter[str] = Counter()
    missing_labels: list[str] = []
    parse_errors: list[str] = []
    length_mismatches: list[str] = []
    sequence_roots = sorted(path for path in split_root.iterdir() if path.is_dir())
    totals["sequences"] = len(sequence_roots)

    for sequence_root in sequence_roots:
        images = _image_names(sequence_root)
        totals["images"] += len(images)
        if not images:
            totals["empty_sequences"] += 1
        numeric_indices = [int(Path(name).stem) for name in images if Path(name).stem.isdigit()]
        if len(numeric_indices) != len(images) or numeric_indices != list(
            range(1, len(images) + 1)
        ):
            totals["noncontiguous_frame_sequences"] += 1

        label_path = sequence_root / "IR_label.json"
        if not label_path.is_file():
            missing_labels.append(sequence_root.name)
            continue
        try:
            labels = _load_json(label_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parse_errors.append(f"{sequence_root.name}: {error}")
            continue
        exists = labels.get("exist")
        boxes = labels.get("gt_rect")
        if not isinstance(exists, list) or not isinstance(boxes, list):
            totals["invalid_core_arrays"] += 1
            continue
        if len(images) != len(exists) or len(images) != len(boxes):
            length_mismatches.append(
                f"{sequence_root.name}: images={len(images)}, "
                f"exist={len(exists)}, gt_rect={len(boxes)}"
            )

        for key in ANTIUAV410_ATTRIBUTE_KEYS:
            values = labels.get(key)
            if values is None:
                attribute_missing[key] += 1
            elif not isinstance(values, list) or len(values) != len(images):
                attribute_incomplete[key] += 1

        for present, source_box in zip(exists, boxes, strict=False):
            if present not in (0, 1):
                totals["invalid_exist_values"] += 1
                continue
            box, reason = _box_status(source_box)
            if present == 1:
                totals["present_frames"] += 1
                if reason is None:
                    totals["valid_present_boxes"] += 1
                else:
                    totals["invalid_present_boxes"] += 1
                    totals[f"invalid_present_{reason}"] += 1
            else:
                totals["absent_frames"] += 1
                if _box_is_nonzero(source_box):
                    totals["absent_frames_with_nonzero_box"] += 1

        if images:
            try:
                with Image.open(sequence_root / images[0]) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
                    totals[f"sample_image_mode_{image.mode}"] += 1
            except (OSError, ValueError) as error:
                totals["sample_image_open_failures"] += 1
                parse_errors.append(f"{sequence_root.name}/{images[0]}: {error}")

    return {
        **dict(totals),
        "missing_split": False,
        "missing_label_files": missing_labels,
        "label_parse_errors": parse_errors,
        "frame_label_length_mismatches": length_mismatches,
        "sampled_sequence_dimensions": dict(sorted(dimensions.items())),
        "missing_attribute_arrays": dict(sorted(attribute_missing.items())),
        "incomplete_attribute_arrays": dict(sorted(attribute_incomplete.items())),
    }


def _archive_report(archive: Path, *, verify_crc: bool) -> dict:
    split_sequences = {split: set() for split in ANTIUAV410_SPLITS}
    split_images: Counter[str] = Counter()
    split_labels: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for member in members:
            parts = PurePosixPath(member.filename).parts
            if len(parts) < 3 or parts[0] not in ANTIUAV410_SPLITS:
                continue
            split, sequence, filename = parts[0], parts[1], parts[2]
            if not sequence:
                continue
            split_sequences[split].add(sequence)
            split_images[split] += int(filename.lower().endswith(".jpg"))
            split_labels[split] += int(filename == "IR_label.json")
        bad_member = bundle.testzip() if verify_crc else None
    return {
        "name": Path(archive).name,
        "bytes": Path(archive).stat().st_size,
        "sha256": sha256_file(Path(archive)),
        "members": len(members),
        "crc_checked": verify_crc,
        "first_bad_crc_member": bad_member,
        "split_sequences": {split: len(sequences) for split, sequences in split_sequences.items()},
        "split_images": {split: split_images[split] for split in ANTIUAV410_SPLITS},
        "split_label_files": {split: split_labels[split] for split in ANTIUAV410_SPLITS},
    }


def _split_layout_ok(report: dict) -> bool:
    return not bool(
        report.get("missing_split")
        or report.get("empty_sequences")
        or report.get("noncontiguous_frame_sequences")
        or report.get("invalid_core_arrays")
        or report.get("invalid_exist_values")
        or report.get("sample_image_open_failures")
        or report.get("missing_label_files")
        or report.get("label_parse_errors")
        or report.get("frame_label_length_mismatches")
    )


def audit_release(
    root: Path,
    *,
    archive: Path | None = None,
    verify_archive_crc: bool = True,
) -> dict:
    """Audit Anti-UAV410 as an external IR-only evaluation source."""
    root = Path(root)
    splits = {split: audit_split(root, split) for split in ANTIUAV410_SPLITS}
    sequence_sets = {
        split: {path.name for path in (root / split).iterdir() if path.is_dir()}
        if (root / split).is_dir()
        else set()
        for split in ANTIUAV410_SPLITS
    }
    overlap = {
        "train_val": sorted(sequence_sets["train"] & sequence_sets["val"]),
        "train_test": sorted(sequence_sets["train"] & sequence_sets["test"]),
        "val_test": sorted(sequence_sets["val"] & sequence_sets["test"]),
    }
    archive_report = (
        _archive_report(Path(archive), verify_crc=verify_archive_crc) if archive else None
    )
    extracted_counts = {
        "split_sequences": {
            split: splits[split].get("sequences", 0) for split in ANTIUAV410_SPLITS
        },
        "split_images": {split: splits[split].get("images", 0) for split in ANTIUAV410_SPLITS},
        "split_label_files": {
            split: splits[split].get("sequences", 0)
            - len(splits[split].get("missing_label_files", []))
            for split in ANTIUAV410_SPLITS
        },
    }
    archive_ok = bool(
        archive_report
        and archive_report["crc_checked"]
        and archive_report["first_bad_crc_member"] is None
        and all(
            archive_report[key] == extracted_counts[key]
            for key in ("split_sequences", "split_images", "split_label_files")
        )
    )
    extraction_ok = all(_split_layout_ok(report) for report in splits.values())
    dimensions_ok = all(
        report.get("sampled_sequence_dimensions")
        == {f"{ANTIUAV410_WIDTH}x{ANTIUAV410_HEIGHT}": report.get("sequences", 0)}
        for report in splits.values()
    )
    image_modes_ok = all(
        report.get(f"sample_image_mode_{ANTIUAV410_IMAGE_MODE}", 0) == report.get("sequences", 0)
        for report in splits.values()
    )
    overlap_ok = not any(overlap.values())
    annotation_anomalies = sum(
        report.get("invalid_present_boxes", 0) + report.get("absent_frames_with_nonzero_box", 0)
        for report in splits.values()
    )
    attributes_complete = not any(
        report.get("missing_attribute_arrays") or report.get("incomplete_attribute_arrays")
        for report in splits.values()
    )
    external_ok = bool(
        archive_ok
        and extraction_ok
        and dimensions_ok
        and image_modes_ok
        and overlap_ok
        and splits["test"].get("valid_present_boxes", 0) > 0
    )
    return {
        "dataset": "Anti-UAV410",
        "root": str(root.resolve()),
        "representation": "display-referred 8-bit thermal JPEG stored as RGB",
        "role": "IR-only external evaluation; prohibited for generation and curation",
        "source_archive": archive_report,
        "splits": splits,
        "sequence_overlap": overlap,
        "gates": {
            "archive_integrity": "pass" if archive_ok else "hold",
            "extraction_completeness": "pass" if extraction_ok and dimensions_ok else "hold",
            "image_storage_contract": "pass" if image_modes_ok else "hold",
            "official_split_sequence_disjointness": "pass" if overlap_ok else "hold",
            "annotation_integrity": "pass" if annotation_anomalies == 0 else "hold",
            "attribute_completeness": "pass" if attributes_complete else "hold",
            "external_evaluation_eligible": "pass" if external_ok else "hold",
            "generator_or_curation_eligible": "hold",
        },
        "exclusion_policy": {
            "invalid_present_frame": "exclude the entire frame and count the reason",
            "absent_frame_with_nonzero_box": "retain as a negative; existence flag is canonical",
            "absent_frame": "retain as an external-evaluation negative",
        },
        "notes": [
            "Anti-UAV410 is IR-only and cannot supply visible-to-IR generator pairs.",
            "Incomplete attributes are secondary metadata and never replace existence labels.",
            "External test frames must never enter training, generation, or curation.",
        ],
    }


def verify_external_manifest(manifest: dict) -> None:
    """Reject a changed or malformed Anti-UAV410 external-evaluation manifest."""
    if manifest.get("schema_version") != ANTIUAV410_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported Anti-UAV410 manifest schema: {manifest.get('schema_version')}"
        )
    recorded = manifest.get("manifest_sha256")
    if not isinstance(recorded, str):
        raise ValueError("Anti-UAV410 manifest has no manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = _canonical_sha256(unsigned)
    if actual != recorded:
        raise ValueError(f"Anti-UAV410 manifest hash mismatch: expected {recorded}, got {actual}")
    if manifest.get("dataset") != "Anti-UAV410":
        raise ValueError("Anti-UAV410 manifest has the wrong dataset identity")
    if manifest.get("role") != "external_evaluation_only":
        raise ValueError("Anti-UAV410 manifest must be external_evaluation_only")
    if manifest.get("representation") != "display8_thermal_jpeg_rgb":
        raise ValueError("Anti-UAV410 manifest has an unsupported representation")
    image_storage = manifest.get("image_storage", {})
    if image_storage != {
        "format": "JPEG",
        "mode": ANTIUAV410_IMAGE_MODE,
        "channels": ANTIUAV410_CHANNELS,
        "width": ANTIUAV410_WIDTH,
        "height": ANTIUAV410_HEIGHT,
    }:
        raise ValueError("Anti-UAV410 manifest has an unsupported image storage contract")
    if manifest.get("categories") != [{"name": "uav", "source_category_id": 1, "class_index": 0}]:
        raise ValueError("Anti-UAV410 manifest has an unsupported category mapping")
    if manifest.get("target_pixel_area_bins") != {
        "edges": list(TARGET_AREA_BIN_EDGES),
        "names": list(TARGET_AREA_BIN_NAMES),
        "units": "pixels_squared",
    }:
        raise ValueError("Anti-UAV410 manifest changed the frozen target-area bins")
    if set(manifest.get("splits", {})) != {"test"}:
        raise ValueError("Anti-UAV410 external manifest must contain only the test split")
    records = manifest["splits"]["test"].get("records", [])
    if not isinstance(records, list):
        raise ValueError("Anti-UAV410 manifest records must be a list")
    image_ids = [record.get("image_id") for record in records]
    if image_ids != list(range(1, len(records) + 1)):
        raise ValueError("Anti-UAV410 manifest image ids must be contiguous from one")
    annotation_ids = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Anti-UAV410 manifest records must be objects")
        path = PurePosixPath(str(record.get("image_path", "")))
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 3
            or path.suffix.lower() != ".jpg"
        ):
            raise ValueError(f"invalid Anti-UAV410 image path: {path}")
        if path.parts[0] != "test" or path.parts[1] != record.get("sequence_id"):
            raise ValueError(f"Anti-UAV410 path/sequence mismatch: {path}")
        if record.get("width") != ANTIUAV410_WIDTH or record.get("height") != ANTIUAV410_HEIGHT:
            raise ValueError(f"Anti-UAV410 frame dimensions changed: {path}")
        try:
            source_frame_index = int(path.stem)
        except ValueError as error:
            raise ValueError(f"Anti-UAV410 frame name is not numeric: {path}") from error
        if source_frame_index != record.get("frame_index"):
            raise ValueError(f"Anti-UAV410 frame index/path mismatch: {path}")
        annotations = record.get("annotations", [])
        if not isinstance(annotations, list) or len(annotations) > 1:
            raise ValueError(f"Anti-UAV410 frame must have zero or one annotation: {path}")
        visibility = record.get("visibility")
        if visibility not in {"present", "absent"}:
            raise ValueError(f"Anti-UAV410 visibility is invalid: {path}")
        visible = visibility == "present"
        if visible != bool(annotations):
            raise ValueError(f"Anti-UAV410 visibility/annotation mismatch: {path}")
        if record.get("source_exist") != int(visible):
            raise ValueError(f"Anti-UAV410 existence/visibility mismatch: {path}")
        attributes = record.get("attributes")
        if not isinstance(attributes, dict) or set(attributes) != set(ANTIUAV410_ATTRIBUTE_KEYS):
            raise ValueError(f"Anti-UAV410 attributes do not match the frozen schema: {path}")
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError(f"Anti-UAV410 annotations must be objects: {path}")
            box, reason = _box_status(annotation.get("bbox_xywh"))
            if reason is not None or box is None:
                raise ValueError(f"invalid Anti-UAV410 box in manifest: {path}")
            if annotation.get("source_category_id") != 1 or annotation.get("class_index") != 0:
                raise ValueError(f"invalid Anti-UAV410 category mapping: {path}")
            area = box[2] * box[3]
            if annotation.get("area") != area or record.get("target_pixel_area") != area:
                raise ValueError(f"Anti-UAV410 area/box mismatch: {path}")
            if record.get("target_pixel_area_bin") != target_pixel_area_bin(area):
                raise ValueError(f"Anti-UAV410 area-bin mismatch: {path}")
            annotation_ids.append(annotation.get("annotation_id"))
        if not visible and (
            record.get("target_pixel_area") is not None
            or record.get("target_pixel_area_bin") != "absent"
        ):
            raise ValueError(f"Anti-UAV410 negative frame has target area metadata: {path}")
    if annotation_ids != list(range(1, len(annotation_ids) + 1)):
        raise ValueError("Anti-UAV410 annotation ids must be contiguous from one")


def build_external_manifest(root: Path, *, audit_report: dict) -> dict:
    """Freeze the Anti-UAV410 test split for external frame-level detection evaluation."""
    root = Path(root)
    audit_root = Path(audit_report.get("root", ""))
    if audit_root.resolve() != root.resolve():
        raise ValueError(f"audit root {audit_root} does not match dataset root {root}")
    gate = audit_report.get("gates", {}).get("external_evaluation_eligible")
    if gate != "pass":
        raise ValueError(f"external_evaluation_eligible must pass, got {gate!r}")

    records: list[dict] = []
    source_annotations: list[dict] = []
    excluded: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    annotation_id = 1
    image_id = 1
    split_root = root / "test"
    for sequence_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
        label_path = sequence_root / "IR_label.json"
        labels = _load_json(label_path)
        images = _image_names(sequence_root)
        exists = labels.get("exist")
        boxes = labels.get("gt_rect")
        if not isinstance(exists, list) or not isinstance(boxes, list):
            raise ValueError(f"invalid core annotation arrays in {label_path}")
        if len(images) != len(exists) or len(images) != len(boxes):
            raise ValueError(f"frame/annotation length mismatch in {label_path}")
        source_annotations.append(
            {
                "sequence_id": sequence_root.name,
                "path": f"test/{sequence_root.name}/IR_label.json",
                "sha256": sha256_file(label_path),
                "frames": len(images),
            }
        )

        for frame_offset, (filename, present, source_box) in enumerate(
            zip(images, exists, boxes, strict=True),
            start=1,
        ):
            counts["source_frames"] += 1
            if present not in (0, 1):
                raise ValueError(
                    f"invalid existence value in {sequence_root.name} frame {frame_offset}"
                )
            box, reason = _box_status(source_box)
            if present == 1 and reason is not None:
                excluded[reason] += 1
                continue

            attributes = {}
            for key in ANTIUAV410_ATTRIBUTE_KEYS:
                values = labels.get(key)
                attributes[key] = (
                    values[frame_offset - 1]
                    if isinstance(values, list) and frame_offset <= len(values)
                    else None
                )
            annotations = []
            area = None
            area_bin = "absent"
            if present == 1 and box is not None:
                area = float(box[2] * box[3])
                area_bin = target_pixel_area_bin(area)
                annotations.append(
                    {
                        "annotation_id": annotation_id,
                        "source_category_id": 1,
                        "class_index": 0,
                        "bbox_xywh": box,
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
                counts["positive_images"] += 1
            else:
                counts["negative_images"] += 1
                if _box_is_nonzero(source_box):
                    counts["absent_frames_with_nonzero_source_box"] += 1

            records.append(
                {
                    "image_id": image_id,
                    "image_path": f"test/{sequence_root.name}/{filename}",
                    "width": ANTIUAV410_WIDTH,
                    "height": ANTIUAV410_HEIGHT,
                    "sequence_id": sequence_root.name,
                    "frame_index": frame_offset,
                    "source_exist": int(present),
                    "visibility": "present" if present == 1 else "absent",
                    "target_pixel_area": area,
                    "target_pixel_area_bin": area_bin,
                    "attributes": attributes,
                    "annotations": annotations,
                }
            )
            image_id += 1

    counts["included_images"] = len(records)
    counts["annotations"] = annotation_id - 1
    counts["excluded_invalid_present_images"] = sum(excluded.values())
    manifest = {
        "schema_version": ANTIUAV410_MANIFEST_SCHEMA,
        "dataset": "Anti-UAV410",
        "role": "external_evaluation_only",
        "representation": "display8_thermal_jpeg_rgb",
        "image_storage": {
            "format": "JPEG",
            "mode": ANTIUAV410_IMAGE_MODE,
            "channels": ANTIUAV410_CHANNELS,
            "width": ANTIUAV410_WIDTH,
            "height": ANTIUAV410_HEIGHT,
        },
        "categories": [{"name": "uav", "source_category_id": 1, "class_index": 0}],
        "target_pixel_area_bins": {
            "edges": list(TARGET_AREA_BIN_EDGES),
            "names": list(TARGET_AREA_BIN_NAMES),
            "units": "pixels_squared",
        },
        "source_archive": audit_report.get("source_archive"),
        "exclusion_policy": audit_report.get("exclusion_policy"),
        "splits": {
            "test": {
                "source_annotations": source_annotations,
                "counts": dict(counts),
                "exclusion_counts": dict(sorted(excluded.items())),
                "records": records,
            }
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    verify_external_manifest(manifest)
    return manifest


def export_external_coco(manifest_path: Path, output_path: Path) -> dict:
    """Export the frozen test manifest as COCO without copying source imagery."""
    manifest = _load_json(Path(manifest_path))
    verify_external_manifest(manifest)
    records = manifest["splits"]["test"]["records"]
    images = []
    annotations = []
    for record in records:
        images.append(
            {
                "id": record["image_id"],
                "file_name": record["image_path"],
                "width": record["width"],
                "height": record["height"],
                "video_id": record["sequence_id"],
                "frame_id": record["frame_index"],
                "visibility": record["visibility"],
                "target_pixel_area": record["target_pixel_area"],
                "target_pixel_area_bin": record["target_pixel_area_bin"],
                "attributes": record["attributes"],
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
            "dataset": manifest["dataset"],
            "role": manifest["role"],
            "representation": manifest["representation"],
            "exclusion_policy": manifest["exclusion_policy"],
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "uav", "supercategory": "aircraft"}],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "images": len(images),
        "annotations": len(annotations),
    }
