"""Manifest-locked native-IR engineering subset for Anti-UAV300 detection."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

from aero_ir.data.antiuav import load_split_manifest, probe_video
from aero_ir.data.antiuav410 import target_pixel_area_bin
from aero_ir.data.flir import sha256_file
from aero_ir.utils.manifest import canonical_hash, file_sha256

ANTIUAV300_IR_MANIFEST_SCHEMA = 1
ANTIUAV300_IR_SPLITS = ("train", "val")
ANTIUAV300_IR_SAMPLES_PER_SEQUENCE = {"train": 8, "val": 4}


def uniform_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Select a fixed number of endpoint-inclusive, label-independent frame indices."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not 1 <= sample_count <= frame_count:
        raise ValueError("sample_count must be between one and frame_count")
    if sample_count == 1:
        return [frame_count // 2]
    return [index * (frame_count - 1) // (sample_count - 1) for index in range(sample_count)]


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _box_status(box: object, width: int, height: int) -> tuple[list[float] | None, str | None]:
    if not isinstance(box, list) or len(box) != 4:
        return None, "malformed"
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None, "non_numeric"
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite"
    x, y, box_width, box_height = values
    if box_width <= 0 or box_height <= 0:
        return values, "non_positive_extent"
    if x < 0 or y < 0 or x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
        return values, "out_of_bounds"
    return values, None


def _box_is_nonzero(box: object) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        return any(float(value) != 0.0 for value in box)
    except (TypeError, ValueError):
        return False


def _check_source_audit(root: Path, audit_report: dict) -> None:
    audit_root = Path(str(audit_report.get("root", "")))
    if audit_root.resolve() != root.resolve():
        raise ValueError(f"audit root {audit_root} does not match dataset root {root}")
    required = {
        "archive_integrity": "pass",
        "extraction_completeness": "pass",
        "official_manifest_sequence_disjointness": "pass",
        "paired_temporal_integrity": "pass",
        "train_only_pilot_eligible": "pass",
    }
    gates = audit_report.get("gates", {})
    failures = {
        name: gates.get(name) for name, expected in required.items() if gates.get(name) != expected
    }
    if failures:
        raise ValueError(f"Anti-UAV300 source audit does not qualify the IR-only pilot: {failures}")


def _decode_selected_frames(
    video_path: Path,
    selected_indices: list[int],
    destinations: dict[int, Path],
    *,
    width: int,
    height: int,
) -> dict[int, dict[str, str | int]]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - minimal environments
        raise RuntimeError("Anti-UAV300 IR preparation requires the 'detect' extra") from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open {video_path}")
    decoded: dict[int, dict[str, str | int]] = {}
    try:
        for frame_index in selected_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"could not decode {video_path} frame {frame_index}")
            if frame.dtype.name != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"unexpected decoded frame contract in {video_path} frame {frame_index}: "
                    f"shape={frame.shape}, dtype={frame.dtype}"
                )
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"decoded dimensions changed in {video_path} frame {frame_index}: "
                    f"{frame.shape[1]}x{frame.shape[0]} != {width}x{height}"
                )
            destination = destinations[frame_index]
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded_ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not encoded_ok:
                raise ValueError(f"could not encode {destination}")
            destination.write_bytes(encoded.tobytes())
            decoded[frame_index] = {
                "prepared_image_sha256": file_sha256(destination),
                "prepared_image_bytes": destination.stat().st_size,
            }
    finally:
        capture.release()
    return decoded


def _opencv_version() -> str:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - minimal environments
        raise RuntimeError("Anti-UAV300 IR preparation requires the 'detect' extra") from error
    return str(cv2.__version__)


def prepare_native_ir_smoke(
    root: Path,
    output_root: Path,
    *,
    audit_report: dict,
    samples_per_sequence: dict[str, int] | None = None,
) -> dict:
    """Decode a label-independent IR subset and freeze its provenance and exclusions."""
    root = Path(root).resolve()
    output_root = Path(output_root).resolve()
    _check_source_audit(root, audit_report)
    sample_counts = dict(ANTIUAV300_IR_SAMPLES_PER_SEQUENCE)
    if samples_per_sequence is not None:
        sample_counts = dict(samples_per_sequence)
    if set(sample_counts) != set(ANTIUAV300_IR_SPLITS):
        raise ValueError("samples_per_sequence must define exactly train and val")
    if any(not isinstance(value, int) or value <= 0 for value in sample_counts.values()):
        raise ValueError("samples_per_sequence values must be positive integers")

    splits: dict[str, dict] = {}
    next_image_id = {split: 1 for split in ANTIUAV300_IR_SPLITS}
    next_annotation_id = {split: 1 for split in ANTIUAV300_IR_SPLITS}
    for split in ANTIUAV300_IR_SPLITS:
        official_manifest_path = root / "label_new" / f"{split}.json"
        official_manifest = load_split_manifest(root, split)
        records: list[dict] = []
        exclusions: list[dict] = []
        source_annotations: list[dict] = []
        counts: Counter[str] = Counter()
        for sequence_id in sorted(official_manifest):
            sequence_root = root / split / sequence_id
            label_path = sequence_root / "infrared.json"
            video_path = sequence_root / "infrared.mp4"
            media = probe_video(video_path)
            if not media["opened"]:
                raise ValueError(f"could not open infrared video in {sequence_id}")
            selected = uniform_frame_indices(media["frames"], sample_counts[split])
            labels = _load_object(label_path)
            exists = labels.get("exist")
            boxes = labels.get("gt_rect")
            if (
                not isinstance(exists, list)
                or not isinstance(boxes, list)
                or len(exists) != len(boxes)
            ):
                raise ValueError(f"invalid annotation arrays in {label_path}")
            if media["frames"] != len(exists):
                raise ValueError(f"video/annotation mismatch in {sequence_id}")
            width = int(media["width"])
            height = int(media["height"])
            source_annotations.append(
                {
                    "sequence_id": sequence_id,
                    "path": f"{split}/{sequence_id}/infrared.json",
                    "sha256": sha256_file(label_path),
                    "frames": len(exists),
                    "selected_frame_indices_zero_based": selected,
                }
            )
            destinations: dict[int, Path] = {}
            pending: list[dict] = []
            for frame_index in selected:
                counts["selected_source_frames"] += 1
                present = exists[frame_index]
                source_box = boxes[frame_index]
                if present not in (0, 1):
                    raise ValueError(
                        f"invalid existence value in {sequence_id} frame {frame_index}"
                    )
                box, reason = _box_status(source_box, width, height)
                if present == 1 and reason is not None:
                    exclusions.append(
                        {
                            "sequence_id": sequence_id,
                            "frame_index_zero_based": frame_index,
                            "reason": reason,
                            "source_box": source_box,
                        }
                    )
                    counts[f"excluded_{reason}"] += 1
                    continue

                relative_image = PurePosixPath(
                    "images", split, sequence_id, f"frame_{frame_index:06d}.png"
                )
                destination = output_root / Path(relative_image)
                destinations[frame_index] = destination
                annotations = []
                area = None
                area_bin = "absent"
                if present == 1 and box is not None:
                    area = float(box[2] * box[3])
                    area_bin = target_pixel_area_bin(area)
                    annotations.append(
                        {
                            "annotation_id": next_annotation_id[split],
                            "source_category_id": 1,
                            "class_index": 0,
                            "bbox_xywh": box,
                            "area": area,
                            "iscrowd": 0,
                        }
                    )
                    next_annotation_id[split] += 1
                    counts["positive_images"] += 1
                else:
                    counts["negative_images"] += 1
                    if _box_is_nonzero(source_box):
                        counts["absent_frames_with_nonzero_source_box"] += 1
                pending.append(
                    {
                        "image_id": next_image_id[split],
                        "image_path": relative_image.as_posix(),
                        "width": width,
                        "height": height,
                        "sequence_id": sequence_id,
                        "sequence_attributes": official_manifest[sequence_id],
                        "frame_index_zero_based": frame_index,
                        "source_exist": int(present),
                        "visibility": "present" if present == 1 else "absent",
                        "target_pixel_area": area,
                        "target_pixel_area_bin": area_bin,
                        "annotations": annotations,
                    }
                )
                next_image_id[split] += 1

            decoded = _decode_selected_frames(
                video_path,
                sorted(destinations),
                destinations,
                width=width,
                height=height,
            )
            for record in pending:
                record.update(decoded[record["frame_index_zero_based"]])
                records.append(record)

        counts["included_images"] = len(records)
        counts["annotations"] = next_annotation_id[split] - 1
        counts["excluded_invalid_present_images"] = len(exclusions)
        splits[split] = {
            "official_split_manifest": {
                "path": f"label_new/{split}.json",
                "sha256": sha256_file(official_manifest_path),
            },
            "source_annotations": source_annotations,
            "counts": dict(counts),
            "exclusions": exclusions,
            "records": records,
        }

    manifest = {
        "schema_version": ANTIUAV300_IR_MANIFEST_SCHEMA,
        "dataset": "Anti-UAV300",
        "role": "native_ir_detector_engineering_smoke_only",
        "scientific_status": "not_reportable_and_does_not_clear_paired_registration_gate",
        "representation": "display_referred_8bit_ir_video_decoded_to_lossless_png",
        "image_storage": {
            "format": "PNG",
            "decoder": "OpenCV VideoCapture",
            "opencv_version": _opencv_version(),
            "dtype": "uint8",
            "channels": 3,
            "channel_order_in_loader": "BGR",
        },
        "sampling": {
            "method": "endpoint_inclusive_uniform_integer_grid_before_label_inspection",
            "frame_index_origin": 0,
            "samples_per_sequence": sample_counts,
            "sequence_order": "lexicographic",
            "random_seed": None,
        },
        "categories": [{"name": "uav", "source_category_id": 1, "class_index": 0}],
        "source_audit_sha256": canonical_hash(audit_report),
        "source_archive": audit_report.get("source_archive"),
        "exclusion_policy": {
            "selected_present_invalid_box": "exclude image and count reason",
            "selected_absent_frame": "retain as negative; existence flag is canonical",
            "unselected_frame": "not decoded and never inspected for sampling",
        },
        "split_usage": {
            "train": "detector_training_only",
            "val": "detector_validation_only",
            "test": "not_accessed",
        },
        "splits": splits,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_native_ir_manifest(manifest, prepared_root=output_root)
    return manifest


def verify_native_ir_manifest(manifest: dict, *, prepared_root: Path | None = None) -> None:
    """Reject a changed, malformed, or incompletely materialized native-IR manifest."""
    if manifest.get("schema_version") != ANTIUAV300_IR_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported Anti-UAV300 IR schema: {manifest.get('schema_version')}")
    recorded = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not isinstance(recorded, str) or canonical_hash(unsigned) != recorded:
        raise ValueError("Anti-UAV300 IR manifest hash mismatch")
    if manifest.get("dataset") != "Anti-UAV300":
        raise ValueError("Anti-UAV300 IR manifest has the wrong dataset identity")
    if manifest.get("role") != "native_ir_detector_engineering_smoke_only":
        raise ValueError("Anti-UAV300 IR manifest has the wrong role")
    if manifest.get("split_usage") != {
        "train": "detector_training_only",
        "val": "detector_validation_only",
        "test": "not_accessed",
    }:
        raise ValueError("Anti-UAV300 IR split usage changed")
    if manifest.get("sampling", {}).get("method") != (
        "endpoint_inclusive_uniform_integer_grid_before_label_inspection"
    ):
        raise ValueError("Anti-UAV300 IR sampling method changed")
    if set(manifest.get("splits", {})) != set(ANTIUAV300_IR_SPLITS):
        raise ValueError("Anti-UAV300 IR manifest must contain train and val only")

    root = Path(prepared_root).resolve() if prepared_root is not None else None
    expected_prepared_paths: set[Path] = set()
    for split in ANTIUAV300_IR_SPLITS:
        payload = manifest["splits"][split]
        records = payload.get("records")
        exclusions = payload.get("exclusions")
        if not isinstance(records, list) or not isinstance(exclusions, list):
            raise ValueError(f"Anti-UAV300 IR {split} records/exclusions must be lists")
        if [record.get("image_id") for record in records] != list(range(1, len(records) + 1)):
            raise ValueError(f"Anti-UAV300 IR {split} image ids are not contiguous")
        selected_by_sequence: dict[str, set[int]] = {}
        for source in payload.get("source_annotations", []):
            sequence_id = source.get("sequence_id")
            frame_count = source.get("frames")
            selected = source.get("selected_frame_indices_zero_based")
            if not isinstance(sequence_id, str) or not isinstance(frame_count, int):
                raise ValueError(f"invalid Anti-UAV300 IR {split} source annotation")
            expected = uniform_frame_indices(
                frame_count, manifest["sampling"]["samples_per_sequence"][split]
            )
            if selected != expected or sequence_id in selected_by_sequence:
                raise ValueError(f"Anti-UAV300 IR {split} source selection changed")
            selected_by_sequence[sequence_id] = set(expected)
        represented: dict[str, set[int]] = {sequence: set() for sequence in selected_by_sequence}
        annotation_ids: list[int] = []
        positive = 0
        negative = 0
        for record in records:
            path = PurePosixPath(str(record.get("image_path", "")))
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 4
                or path.parts[0] != "images"
                or path.parts[1] != split
                or path.parts[2] != record.get("sequence_id")
                or path.suffix.lower() != ".png"
            ):
                raise ValueError(f"invalid Anti-UAV300 IR image path: {path}")
            expected_name = f"frame_{record.get('frame_index_zero_based'):06d}.png"
            if path.name != expected_name:
                raise ValueError(f"Anti-UAV300 IR frame index/path mismatch: {path}")
            sequence_id = record["sequence_id"]
            frame_index = record["frame_index_zero_based"]
            if (
                sequence_id not in represented
                or frame_index not in selected_by_sequence[sequence_id]
            ):
                raise ValueError(f"Anti-UAV300 IR record is outside the frozen selection: {path}")
            if frame_index in represented[sequence_id]:
                raise ValueError(f"duplicate Anti-UAV300 IR selected frame: {path}")
            represented[sequence_id].add(frame_index)
            width = record.get("width")
            height = record.get("height")
            if (
                not isinstance(width, int)
                or width <= 0
                or not isinstance(height, int)
                or height <= 0
            ):
                raise ValueError(f"invalid Anti-UAV300 IR dimensions: {path}")
            annotations = record.get("annotations")
            if not isinstance(annotations, list) or len(annotations) > 1:
                raise ValueError(f"invalid Anti-UAV300 IR annotations: {path}")
            if bool(annotations) != (record.get("source_exist") == 1):
                raise ValueError(f"Anti-UAV300 IR existence/annotation mismatch: {path}")
            if annotations:
                positive += 1
                annotation = annotations[0]
                box, reason = _box_status(annotation.get("bbox_xywh"), width, height)
                if reason is not None or box is None:
                    raise ValueError(f"invalid Anti-UAV300 IR manifest box: {path}")
                area = box[2] * box[3]
                if annotation.get("area") != area or record.get("target_pixel_area") != area:
                    raise ValueError(f"Anti-UAV300 IR box/area mismatch: {path}")
                if record.get("target_pixel_area_bin") != target_pixel_area_bin(area):
                    raise ValueError(f"Anti-UAV300 IR area bin mismatch: {path}")
                if annotation.get("source_category_id") != 1 or annotation.get("class_index") != 0:
                    raise ValueError(f"Anti-UAV300 IR category mapping changed: {path}")
                annotation_ids.append(annotation.get("annotation_id"))
            else:
                negative += 1
                if record.get("target_pixel_area") is not None:
                    raise ValueError(f"Anti-UAV300 IR negative has target area: {path}")
            digest = record.get("prepared_image_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Anti-UAV300 IR prepared image has no digest: {path}")
            if root is not None:
                image_path = root / Path(path)
                expected_prepared_paths.add(image_path)
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if file_sha256(image_path) != digest:
                    raise ValueError(f"Anti-UAV300 IR prepared image changed: {path}")
                if image_path.stat().st_size != record.get("prepared_image_bytes"):
                    raise ValueError(f"Anti-UAV300 IR prepared image size changed: {path}")
        for exclusion in exclusions:
            sequence_id = exclusion.get("sequence_id")
            frame_index = exclusion.get("frame_index_zero_based")
            if (
                sequence_id not in represented
                or frame_index not in selected_by_sequence[sequence_id]
            ):
                raise ValueError(f"Anti-UAV300 IR {split} exclusion is outside the selection")
            if frame_index in represented[sequence_id]:
                raise ValueError(f"duplicate Anti-UAV300 IR {split} selected frame")
            represented[sequence_id].add(frame_index)
        if any(
            represented[sequence] != selected for sequence, selected in selected_by_sequence.items()
        ):
            raise ValueError(f"Anti-UAV300 IR {split} selected-frame accounting changed")
        if annotation_ids != list(range(1, len(annotation_ids) + 1)):
            raise ValueError(f"Anti-UAV300 IR {split} annotation ids are not contiguous")
        counts = payload.get("counts", {})
        expected_counts = {
            "included_images": len(records),
            "annotations": len(annotation_ids),
            "excluded_invalid_present_images": len(exclusions),
            "positive_images": positive,
            "negative_images": negative,
        }
        if any(counts.get(key) != value for key, value in expected_counts.items()):
            raise ValueError(f"Anti-UAV300 IR {split} summary counts changed")
    if root is not None:
        actual_prepared_paths = set((root / "images").rglob("*.png"))
        if actual_prepared_paths != expected_prepared_paths:
            raise ValueError("Anti-UAV300 IR prepared image set contains missing or stale files")


def export_native_ir_coco(manifest_path: Path, output_root: Path) -> dict[str, dict]:
    """Export the frozen prepared subset as standard COCO train/validation files."""
    manifest = _load_object(Path(manifest_path))
    output_root = Path(output_root).resolve()
    verify_native_ir_manifest(manifest, prepared_root=output_root)
    report: dict[str, dict] = {}
    annotation_root = output_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    for split in ANTIUAV300_IR_SPLITS:
        records = manifest["splits"][split]["records"]
        images = [
            {
                "id": record["image_id"],
                "file_name": record["image_path"],
                "width": record["width"],
                "height": record["height"],
                "video_id": record["sequence_id"],
                "frame_id": record["frame_index_zero_based"],
                "visibility": record["visibility"],
                "target_pixel_area": record["target_pixel_area"],
                "target_pixel_area_bin": record["target_pixel_area_bin"],
            }
            for record in records
        ]
        annotations = [
            {
                "id": annotation["annotation_id"],
                "image_id": record["image_id"],
                "category_id": annotation["source_category_id"],
                "bbox": annotation["bbox_xywh"],
                "area": annotation["area"],
                "iscrowd": annotation["iscrowd"],
            }
            for record in records
            for annotation in record["annotations"]
        ]
        payload = {
            "info": {
                "aero_manifest_sha256": manifest["manifest_sha256"],
                "dataset": manifest["dataset"],
                "role": manifest["role"],
                "representation": manifest["representation"],
                "split": split,
            },
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "uav", "supercategory": "aircraft"}],
        }
        destination = annotation_root / f"{split}.json"
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report[split] = {
            "path": str(destination),
            "sha256": file_sha256(destination),
            "images": len(images),
            "annotations": len(annotations),
        }
    return report


def verify_prepared_native_ir_smoke(output_root: Path) -> dict:
    """Verify all prepared frames plus the content-addressed COCO preflight record."""
    output_root = Path(output_root).resolve()
    manifest = _load_object(output_root / "manifest.json")
    verify_native_ir_manifest(manifest, prepared_root=output_root)
    preflight = _load_object(output_root / "preflight.json")
    recorded = preflight.get("preflight_sha256")
    unsigned = {key: value for key, value in preflight.items() if key != "preflight_sha256"}
    if not isinstance(recorded, str) or canonical_hash(unsigned) != recorded:
        raise ValueError("Anti-UAV300 IR preflight hash mismatch")
    if preflight.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Anti-UAV300 IR preflight/manifest mismatch")
    if preflight.get("gate") != "pass" or preflight.get("gpu_used") is not False:
        raise ValueError("Anti-UAV300 IR CPU preflight did not pass")
    annotations = preflight.get("annotations", {})
    if set(annotations) != set(ANTIUAV300_IR_SPLITS):
        raise ValueError("Anti-UAV300 IR preflight annotation splits changed")
    for split in ANTIUAV300_IR_SPLITS:
        specification = annotations[split]
        path = output_root / "annotations" / f"{split}.json"
        if path.resolve() != Path(specification.get("path", "")).resolve():
            raise ValueError(f"Anti-UAV300 IR {split} annotation path changed")
        if file_sha256(path) != specification.get("sha256"):
            raise ValueError(f"Anti-UAV300 IR {split} COCO annotation changed")
        payload = _load_object(path)
        if payload.get("info", {}).get("aero_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"Anti-UAV300 IR {split} COCO/manifest mismatch")
        if len(payload.get("images", [])) != specification.get("images"):
            raise ValueError(f"Anti-UAV300 IR {split} COCO image count changed")
        if len(payload.get("annotations", [])) != specification.get("annotations"):
            raise ValueError(f"Anti-UAV300 IR {split} COCO annotation count changed")
    return preflight
