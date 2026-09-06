"""Immutable FLIR video-pair manifest and conservative registration audit."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from aero_ir.utils.manifest import canonical_hash, file_sha256

PAIR_MANIFEST_SCHEMA = 1
REGISTRATION_AUDIT_SCHEMA = 1
_FRAME_NAME = re.compile(r"^video-([^-]+)-frame-(\d+)-[^/]+\.jpg$")


def _load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_index(coco: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for image in coco.get("images", []):
        name = Path(image["file_name"]).name
        if name in result:
            raise ValueError(f"duplicate COCO image basename: {name}")
        result[name] = image
    return result


def _frame_identity(name: str) -> tuple[str, int]:
    match = _FRAME_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"unrecognized FLIR frame name: {name}")
    return match.group(1), int(match.group(2))


def _annotation_index(coco: dict) -> tuple[dict[int, dict[tuple[int, int], dict]], int]:
    grouped: dict[int, dict[tuple[int, int], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for annotation in coco.get("annotations", []):
        track_id = annotation.get("track_id")
        if track_id is None:
            continue
        key = (int(track_id), int(annotation["category_id"]))
        grouped[int(annotation["image_id"])][key].append(annotation)

    duplicate_keys = 0
    unique: dict[int, dict[tuple[int, int], dict]] = {}
    for image_id, by_key in grouped.items():
        unique[image_id] = {}
        for key, matching_annotations in by_key.items():
            if len(matching_annotations) == 1:
                unique[image_id][key] = matching_annotations[0]
            else:
                duplicate_keys += 1
    return unique, duplicate_keys


def _normalized_bbox(annotation: dict, image: dict) -> np.ndarray:
    x, y, width, height = (float(value) for value in annotation["bbox"])
    return np.asarray(
        [
            x / float(image["width"]),
            y / float(image["height"]),
            width / float(image["width"]),
            height / float(image["height"]),
        ],
        dtype=np.float64,
    )


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def audit_video_pair_registration(
    root: str | Path,
    map_path: str | Path,
    *,
    max_center_residual_p95: float = 0.02,
    min_median_iou: float = 0.5,
) -> dict:
    """Audit direct normalized box reuse using unambiguous shared track/category keys.

    Track identifiers are not assumed to correspond across modalities. A comparison is admitted
    only when the same unique ``(track_id, category_id)`` occurs in an officially paired frame.
    Sequence coverage is reported so a coincidental match in one video cannot qualify a global
    transform.
    """
    root = Path(root).resolve()
    map_path = Path(map_path).resolve()
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("FLIR video-pair map must be a JSON object")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("FLIR video-pair map contains duplicate thermal targets")
    rgb_coco_path = root / "video_rgb_test" / "coco.json"
    thermal_coco_path = root / "video_thermal_test" / "coco.json"
    rgb_coco = _load_coco(rgb_coco_path)
    thermal_coco = _load_coco(thermal_coco_path)
    rgb_images = _image_index(rgb_coco)
    thermal_images = _image_index(thermal_coco)
    rgb_annotations, rgb_duplicate_keys = _annotation_index(rgb_coco)
    thermal_annotations, thermal_duplicate_keys = _annotation_index(thermal_coco)

    centers: list[float] = []
    ious: list[float] = []
    sequence_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    rgb_annotation_total = 0
    thermal_annotation_total = 0
    unmatched_rgb = 0
    unmatched_thermal = 0

    for rgb_name, thermal_name in sorted(mapping.items()):
        rgb_image = rgb_images.get(rgb_name)
        thermal_image = thermal_images.get(thermal_name)
        if rgb_image is None or thermal_image is None:
            raise ValueError("pair map entry is absent from COCO metadata")
        rgb_video, rgb_frame = _frame_identity(rgb_name)
        thermal_video, thermal_frame = _frame_identity(thermal_name)
        if rgb_frame != thermal_frame:
            raise ValueError(f"paired frame index differs: {rgb_name}, {thermal_name}")
        sequence = (rgb_video, thermal_video)
        rgb_by_key = rgb_annotations.get(int(rgb_image["id"]), {})
        thermal_by_key = thermal_annotations.get(int(thermal_image["id"]), {})
        rgb_keys = set(rgb_by_key)
        thermal_keys = set(thermal_by_key)
        shared_keys = rgb_keys & thermal_keys
        rgb_annotation_total += len(rgb_keys)
        thermal_annotation_total += len(thermal_keys)
        unmatched_rgb += len(rgb_keys - thermal_keys)
        unmatched_thermal += len(thermal_keys - rgb_keys)
        sequence_counts[sequence]["frames"] += 1
        sequence_counts[sequence]["shared_track_category_annotations"] += len(shared_keys)

        for key in sorted(shared_keys):
            rgb_box = _normalized_bbox(rgb_by_key[key], rgb_image)
            thermal_box = _normalized_bbox(thermal_by_key[key], thermal_image)
            rgb_center = rgb_box[:2] + rgb_box[2:] / 2.0
            thermal_center = thermal_box[:2] + thermal_box[2:] / 2.0
            centers.append(float(np.linalg.norm(rgb_center - thermal_center)))
            ious.append(_bbox_iou(rgb_box, thermal_box))

    center_summary = _summary(centers)
    iou_summary = _summary(ious)
    sequences_with_matches = sum(
        counts["shared_track_category_annotations"] > 0 for counts in sequence_counts.values()
    )
    all_sequences_covered = sequences_with_matches == len(sequence_counts) and bool(sequence_counts)
    geometry_pass = bool(
        all_sequences_covered
        and center_summary["p95"] is not None
        and center_summary["p95"] <= max_center_residual_p95
        and iou_summary["p50"] is not None
        and iou_summary["p50"] >= min_median_iou
    )
    report = {
        "schema_version": REGISTRATION_AUDIT_SCHEMA,
        "kind": "flir_official_video_pair_registration_audit",
        "dataset_split": "video_test",
        "inputs": {
            "mapping_sha256": file_sha256(map_path),
            "rgb_coco_sha256": file_sha256(rgb_coco_path),
            "thermal_coco_sha256": file_sha256(thermal_coco_path),
        },
        "thresholds": {
            "max_normalized_center_residual_p95": max_center_residual_p95,
            "min_normalized_bbox_iou_p50": min_median_iou,
            "required_sequence_coverage": 1.0,
        },
        "counts": {
            "official_frame_pairs": len(mapping),
            "sequence_pairs": len(sequence_counts),
            "sequences_with_shared_track_category_keys": sequences_with_matches,
            "shared_track_category_annotations": len(centers),
            "rgb_unique_track_category_annotations": rgb_annotation_total,
            "thermal_unique_track_category_annotations": thermal_annotation_total,
            "unmatched_rgb_annotations": unmatched_rgb,
            "unmatched_thermal_annotations": unmatched_thermal,
            "ambiguous_rgb_track_category_keys": rgb_duplicate_keys,
            "ambiguous_thermal_track_category_keys": thermal_duplicate_keys,
        },
        "sequence_pairs": [
            {
                "rgb_video_id": sequence[0],
                "thermal_video_id": sequence[1],
                **dict(counts),
            }
            for sequence, counts in sorted(sequence_counts.items())
        ],
        "direct_normalized_box_transfer": {
            "center_residual": center_summary,
            "bbox_iou": iou_summary,
            "fraction_iou_at_least_0_5": (
                float(np.mean(np.asarray(ious) >= 0.5)) if ious else None
            ),
        },
        "gates": {
            "registration_evidence": "pass" if centers else "hold",
            "direct_label_transfer": "pass" if geometry_pass else "hold",
            "training_use": "hold",
        },
        "limitations": [
            "The provider documents time-synchronised frame pairs but not cross-modal "
            "track-id semantics.",
            "Only unique shared track/category keys are compared; unmatched annotations "
            "are not guessed.",
            "These pairs belong to video_test and must not calibrate a training transform.",
            "No official RGB-to-thermal mapping is supplied for the train/validation still images.",
        ],
    }
    report["registration_audit_sha256"] = canonical_hash(report)
    verify_registration_audit(report)
    return report


def verify_registration_audit(report: dict) -> None:
    if report.get("schema_version") != REGISTRATION_AUDIT_SCHEMA:
        raise ValueError("unsupported FLIR registration-audit schema")
    recorded = report.get("registration_audit_sha256")
    unsigned = {key: value for key, value in report.items() if key != "registration_audit_sha256"}
    if not isinstance(recorded, str) or canonical_hash(unsigned) != recorded:
        raise ValueError("FLIR registration-audit hash mismatch")
    counts = report.get("counts", {})
    if counts.get("shared_track_category_annotations") != report.get(
        "direct_normalized_box_transfer", {}
    ).get("center_residual", {}).get("count"):
        raise ValueError("FLIR registration-audit count mismatch")
    if report.get("gates", {}).get("training_use") != "hold":
        raise ValueError("FLIR video-test pairs must not be qualified for training")
    if report.get("dataset_split") != "video_test":
        raise ValueError("FLIR registration audit must remain scoped to video_test")


def build_video_pair_manifest(
    root: str | Path,
    map_path: str | Path,
    *,
    registration_audit: dict,
    release_audit: dict | None = None,
) -> dict:
    """Freeze every official RGB/thermal video pair without inventing still-image matches."""
    root = Path(root).resolve()
    map_path = Path(map_path).resolve()
    verify_registration_audit(registration_audit)
    if release_audit is not None:
        if Path(release_audit.get("root", "")).resolve() != root:
            raise ValueError("FLIR release audit refers to a different root")
        if release_audit.get("gates", {}).get("time_synchronised_video_pairs") != "pass":
            raise ValueError("time-synchronised video-pair gate did not pass")

    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("FLIR video-pair map must be a JSON object")
    rgb_coco_path = root / "video_rgb_test" / "coco.json"
    thermal_coco_path = root / "video_thermal_test" / "coco.json"
    rgb_coco = _load_coco(rgb_coco_path)
    thermal_coco = _load_coco(thermal_coco_path)
    rgb_images = _image_index(rgb_coco)
    thermal_images = _image_index(thermal_coco)
    rgb_annotation_counts = Counter(
        int(annotation["image_id"]) for annotation in rgb_coco.get("annotations", [])
    )
    thermal_annotation_counts = Counter(
        int(annotation["image_id"]) for annotation in thermal_coco.get("annotations", [])
    )

    records = []
    sequences: Counter = Counter()
    seen_thermal: set[str] = set()
    for rgb_name, thermal_name in sorted(mapping.items()):
        if Path(rgb_name).name != rgb_name or Path(thermal_name).name != thermal_name:
            raise ValueError("pair map entries must be file basenames")
        if thermal_name in seen_thermal:
            raise ValueError(f"duplicate thermal pair target: {thermal_name}")
        seen_thermal.add(thermal_name)
        if rgb_name not in rgb_images or thermal_name not in thermal_images:
            raise ValueError("pair map entry is absent from COCO metadata")
        rgb_image = rgb_images[rgb_name]
        thermal_image = thermal_images[thermal_name]
        rgb_video, rgb_frame = _frame_identity(rgb_name)
        thermal_video, thermal_frame = _frame_identity(thermal_name)
        if rgb_frame != thermal_frame:
            raise ValueError(f"paired frame index differs: {rgb_name}, {thermal_name}")
        rgb_path = Path("video_rgb_test/data") / rgb_name
        thermal_path = Path("video_thermal_test/data") / thermal_name
        analytics_path = Path("video_thermal_test/analyticsData") / Path(thermal_name).with_suffix(
            ".tiff"
        )
        for path in (rgb_path, thermal_path, analytics_path):
            if not (root / path).is_file():
                raise FileNotFoundError(root / path)
        sequence = (rgb_video, thermal_video)
        sequences[sequence] += 1
        records.append(
            {
                "pair_id": canonical_hash([rgb_name, thermal_name]),
                "frame_index": rgb_frame,
                "rgb": {
                    "image_id": int(rgb_image["id"]),
                    "image_path": rgb_path.as_posix(),
                    "width": int(rgb_image["width"]),
                    "height": int(rgb_image["height"]),
                    "video_id": rgb_video,
                    "annotations": rgb_annotation_counts[int(rgb_image["id"])],
                },
                "thermal": {
                    "image_id": int(thermal_image["id"]),
                    "display_path": thermal_path.as_posix(),
                    "analytics_path": analytics_path.as_posix(),
                    "width": int(thermal_image["width"]),
                    "height": int(thermal_image["height"]),
                    "video_id": thermal_video,
                    "annotations": thermal_annotation_counts[int(thermal_image["id"])],
                },
            }
        )

    manifest = {
        "schema_version": PAIR_MANIFEST_SCHEMA,
        "kind": "flir_official_video_pair_manifest",
        "dataset": "Teledyne FLIR ADAS Thermal Dataset v2",
        "source_split": "video_test",
        "intended_uses": [
            "post-freeze generator evaluation",
            "time-synchronisation and registration diagnostics",
        ],
        "prohibited_uses": [
            "detector or generator training",
            "training-time transform calibration",
            "pairing RGB/thermal train or validation stills by COCO id",
        ],
        "source_archive": release_audit.get("source_archive") if release_audit else None,
        "inputs": {
            "mapping_path": map_path.name,
            "mapping_sha256": file_sha256(map_path),
            "rgb_coco_path": "video_rgb_test/coco.json",
            "rgb_coco_sha256": file_sha256(rgb_coco_path),
            "thermal_coco_path": "video_thermal_test/coco.json",
            "thermal_coco_sha256": file_sha256(thermal_coco_path),
        },
        "counts": {
            "pairs": len(records),
            "sequence_pairs": len(sequences),
            "unique_rgb_images": len({record["rgb"]["image_id"] for record in records}),
            "unique_thermal_images": len({record["thermal"]["image_id"] for record in records}),
        },
        "sequence_pairs": [
            {
                "rgb_video_id": sequence[0],
                "thermal_video_id": sequence[1],
                "frames": count,
            }
            for sequence, count in sorted(sequences.items())
        ],
        "registration_audit": registration_audit,
        "records": records,
    }
    if manifest["counts"]["pairs"] != len(mapping) or len(seen_thermal) != len(mapping):
        raise ValueError("pair manifest is not one-to-one and complete")
    manifest["pair_manifest_sha256"] = canonical_hash(manifest)
    verify_video_pair_manifest(manifest)
    return manifest


def verify_video_pair_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != PAIR_MANIFEST_SCHEMA:
        raise ValueError("unsupported FLIR pair-manifest schema")
    recorded = manifest.get("pair_manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "pair_manifest_sha256"}
    if not isinstance(recorded, str) or canonical_hash(unsigned) != recorded:
        raise ValueError("FLIR pair-manifest hash mismatch")
    if manifest.get("source_split") != "video_test":
        raise ValueError("FLIR official pair manifest must remain scoped to video_test")
    required_prohibitions = {
        "detector or generator training",
        "training-time transform calibration",
        "pairing RGB/thermal train or validation stills by COCO id",
    }
    if not required_prohibitions.issubset(set(manifest.get("prohibited_uses", []))):
        raise ValueError("FLIR pair-manifest training prohibitions are incomplete")
    verify_registration_audit(manifest.get("registration_audit", {}))
    records = manifest.get("records", [])
    if manifest.get("counts", {}).get("pairs") != len(records):
        raise ValueError("FLIR pair-manifest count mismatch")
    pair_ids = [record.get("pair_id") for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("FLIR pair-manifest contains duplicate pair ids")
    rgb_names: set[str] = set()
    thermal_names: set[str] = set()
    for record in records:
        frame_index = record.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
            or not math.isfinite(float(frame_index))
        ):
            raise ValueError("FLIR pair-manifest contains an invalid frame index")
        for key in ("image_path",):
            path = Path(record["rgb"][key])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("FLIR pair-manifest contains an unsafe RGB path")
        for key in ("display_path", "analytics_path"):
            path = Path(record["thermal"][key])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("FLIR pair-manifest contains an unsafe thermal path")
        rgb_name = Path(record["rgb"]["image_path"]).name
        thermal_name = Path(record["thermal"]["display_path"]).name
        rgb_video, rgb_frame = _frame_identity(rgb_name)
        thermal_video, thermal_frame = _frame_identity(thermal_name)
        if frame_index != rgb_frame or frame_index != thermal_frame:
            raise ValueError("FLIR pair-manifest frame identity mismatch")
        if (
            record["rgb"].get("video_id") != rgb_video
            or record["thermal"].get("video_id") != thermal_video
        ):
            raise ValueError("FLIR pair-manifest video identity mismatch")
        if record.get("pair_id") != canonical_hash([rgb_name, thermal_name]):
            raise ValueError("FLIR pair-manifest pair id mismatch")
        if rgb_name in rgb_names or thermal_name in thermal_names:
            raise ValueError("FLIR pair-manifest is not one-to-one")
        rgb_names.add(rgb_name)
        thermal_names.add(thermal_name)
