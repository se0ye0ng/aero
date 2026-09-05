"""Anti-UAV300 metadata, video, and split-integrity helpers."""

from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import numpy as np

from aero_ir.data.flir import sha256_file

ANTIUAV300_SPLITS = ("train", "val", "test")
_REQUIRED_FILES = ("visible.json", "visible.mp4", "infrared.json", "infrared.mp4")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_split_manifest(root: Path, split: str) -> dict[str, list[str]]:
    """Load the supplied sequence-to-attribute manifest for an official split."""
    if split not in ANTIUAV300_SPLITS:
        raise ValueError(f"unknown Anti-UAV300 split: {split}")
    data = _load_json(Path(root) / "label_new" / f"{split}.json")
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in label_new/{split}.json")
    return data


def probe_video(path: Path) -> dict:
    """Read container metadata through OpenCV without decoding the full video."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError("Anti-UAV video checks require the 'detect' extra (OpenCV)") from error

    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        return {
            "opened": opened,
            "frames": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else 0,
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) if opened else 0,
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) if opened else 0,
            "fps": float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0,
        }
    finally:
        capture.release()


def _valid_box(box: object, width: int, height: int) -> tuple[bool, bool]:
    if not isinstance(box, list) or len(box) != 4:
        return False, False
    try:
        x, y, box_width, box_height = (float(value) for value in box)
    except (TypeError, ValueError):
        return False, False
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return False, False
    positive = box_width > 0 and box_height > 0
    in_bounds = (
        positive
        and x >= 0
        and y >= 0
        and x + box_width <= width + 1e-6
        and y + box_height <= height + 1e-6
    )
    return positive, in_bounds


def _audit_annotation(annotation: dict, media: dict) -> dict:
    exists = annotation.get("exist")
    boxes = annotation.get("gt_rect")
    if not isinstance(exists, list) or not isinstance(boxes, list):
        return {
            "frames": 0,
            "present_frames": 0,
            "invalid_exist_values": 1,
            "length_mismatch": 1,
            "invalid_present_boxes": 0,
            "out_of_bounds_present_boxes": 0,
            "absent_frames_with_nonzero_box": 0,
        }

    invalid_exist = sum(value not in (0, 1) for value in exists)
    invalid_boxes = 0
    out_of_bounds = 0
    absent_nonzero = 0
    for present, box in zip(exists, boxes, strict=False):
        positive, in_bounds = _valid_box(box, media["width"], media["height"])
        if present == 1:
            invalid_boxes += int(not positive)
            out_of_bounds += int(positive and not in_bounds)
        elif present == 0 and positive:
            absent_nonzero += 1

    return {
        "frames": len(exists),
        "present_frames": sum(value == 1 for value in exists),
        "invalid_exist_values": invalid_exist,
        "length_mismatch": int(len(exists) != len(boxes)),
        "invalid_present_boxes": invalid_boxes,
        "out_of_bounds_present_boxes": out_of_bounds,
        "absent_frames_with_nonzero_box": absent_nonzero,
    }


def _pair_geometry(visible: dict, infrared: dict, visible_media: dict, ir_media: dict) -> dict:
    visible_exist = visible.get("exist", [])
    infrared_exist = infrared.get("exist", [])
    visible_boxes = visible.get("gt_rect", [])
    infrared_boxes = infrared.get("gt_rect", [])
    n_frames = min(len(visible_exist), len(infrared_exist), len(visible_boxes), len(infrared_boxes))
    presence_mismatches = 0
    centre_errors: list[float] = []
    for index in range(n_frames):
        vis_present = visible_exist[index] == 1
        ir_present = infrared_exist[index] == 1
        presence_mismatches += int(vis_present != ir_present)
        if not (vis_present and ir_present):
            continue
        vis_box = visible_boxes[index]
        ir_box = infrared_boxes[index]
        vis_valid, _ = _valid_box(vis_box, visible_media["width"], visible_media["height"])
        ir_valid, _ = _valid_box(ir_box, ir_media["width"], ir_media["height"])
        if not (vis_valid and ir_valid):
            continue
        vis_x = (float(vis_box[0]) + float(vis_box[2]) / 2) / visible_media["width"]
        vis_y = (float(vis_box[1]) + float(vis_box[3]) / 2) / visible_media["height"]
        ir_x = (float(ir_box[0]) + float(ir_box[2]) / 2) / ir_media["width"]
        ir_y = (float(ir_box[1]) + float(ir_box[3]) / 2) / ir_media["height"]
        centre_errors.append(math.hypot(vis_x - ir_x, vis_y - ir_y))
    return {
        "compared_frames": n_frames,
        "presence_mismatches": presence_mismatches,
        "normalised_centre_errors": centre_errors,
    }


def audit_split(
    root: Path,
    split: str,
    *,
    video_probe: Callable[[Path], dict] = probe_video,
) -> dict:
    """Audit one extracted official split against its supplied manifest."""
    root = Path(root)
    manifest = load_split_manifest(root, split)
    split_root = root / split
    disk_sequences = (
        {path.name for path in split_root.iterdir() if path.is_dir()}
        if split_root.is_dir()
        else set()
    )
    expected_sequences = set(manifest)
    missing_sequences = sorted(expected_sequences - disk_sequences)
    orphan_sequences = sorted(disk_sequences - expected_sequences)

    totals: Counter[str] = Counter()
    dimensions = {"visible": Counter(), "infrared": Counter()}
    fps_values = {"visible": set(), "infrared": set()}
    centre_errors: list[float] = []
    missing_files: list[str] = []
    parse_errors: list[str] = []

    for sequence in sorted(expected_sequences & disk_sequences):
        sequence_root = split_root / sequence
        absent = [name for name in _REQUIRED_FILES if not (sequence_root / name).is_file()]
        missing_files.extend(f"{sequence}/{name}" for name in absent)
        if absent:
            continue
        try:
            visible = _load_json(sequence_root / "visible.json")
            infrared = _load_json(sequence_root / "infrared.json")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parse_errors.append(f"{sequence}: {error}")
            continue

        media = {
            "visible": video_probe(sequence_root / "visible.mp4"),
            "infrared": video_probe(sequence_root / "infrared.mp4"),
        }
        for modality in ("visible", "infrared"):
            info = media[modality]
            totals["media_open_failures"] += int(not info["opened"])
            dimensions[modality][f"{info['width']}x{info['height']}"] += 1
            fps_values[modality].add(round(float(info["fps"]), 6))

        visible_report = _audit_annotation(visible, media["visible"])
        infrared_report = _audit_annotation(infrared, media["infrared"])
        for modality, report in (
            ("visible", visible_report),
            ("infrared", infrared_report),
        ):
            totals.update(report)
            for key in (
                "present_frames",
                "invalid_present_boxes",
                "out_of_bounds_present_boxes",
                "absent_frames_with_nonzero_box",
            ):
                totals[f"{modality}_{key}"] += report[key]
        totals["label_video_frame_mismatches"] += int(
            visible_report["frames"] != media["visible"]["frames"]
        )
        totals["label_video_frame_mismatches"] += int(
            infrared_report["frames"] != media["infrared"]["frames"]
        )
        totals["paired_video_frame_mismatches"] += int(
            media["visible"]["frames"] != media["infrared"]["frames"]
        )
        totals["paired_video_fps_mismatches"] += int(
            not math.isclose(media["visible"]["fps"], media["infrared"]["fps"], abs_tol=1e-6)
        )

        geometry = _pair_geometry(visible, infrared, media["visible"], media["infrared"])
        totals["paired_frames_compared"] += geometry["compared_frames"]
        totals["visible_ir_presence_mismatches"] += geometry["presence_mismatches"]
        centre_errors.extend(geometry["normalised_centre_errors"])
        totals["sequences_audited"] += 1

    centre_array = np.asarray(centre_errors, dtype=np.float64)
    return {
        "manifest_sequences": len(expected_sequences),
        "disk_sequences": len(disk_sequences),
        "sequences_audited": totals["sequences_audited"],
        "missing_sequences": missing_sequences,
        "orphan_sequences": orphan_sequences,
        "missing_required_files": missing_files,
        "annotation_parse_errors": parse_errors,
        "annotation_frames": totals["frames"],
        "present_frame_labels": totals["present_frames"],
        "invalid_exist_values": totals["invalid_exist_values"],
        "annotation_length_mismatches": totals["length_mismatch"],
        "invalid_present_boxes": totals["invalid_present_boxes"],
        "out_of_bounds_present_boxes": totals["out_of_bounds_present_boxes"],
        "absent_frames_with_nonzero_box": totals["absent_frames_with_nonzero_box"],
        "modality_annotation_counts": {
            modality: {
                key: totals[f"{modality}_{key}"]
                for key in (
                    "present_frames",
                    "invalid_present_boxes",
                    "out_of_bounds_present_boxes",
                    "absent_frames_with_nonzero_box",
                )
            }
            for modality in ("visible", "infrared")
        },
        "media_open_failures": totals["media_open_failures"],
        "label_video_frame_mismatches": totals["label_video_frame_mismatches"],
        "paired_video_frame_mismatches": totals["paired_video_frame_mismatches"],
        "paired_video_fps_mismatches": totals["paired_video_fps_mismatches"],
        "video_dimensions": {
            modality: dict(sorted(counter.items())) for modality, counter in dimensions.items()
        },
        "video_fps": {modality: sorted(values) for modality, values in fps_values.items()},
        "paired_frames_compared": totals["paired_frames_compared"],
        "visible_ir_presence_mismatches": totals["visible_ir_presence_mismatches"],
        "paired_present_frames_with_valid_boxes": len(centre_errors),
        "normalised_box_centre_error_median": (
            float(np.median(centre_array)) if centre_array.size else None
        ),
        "normalised_box_centre_error_p95": (
            float(np.percentile(centre_array, 95)) if centre_array.size else None
        ),
    }


def _archive_report(archive: Path, *, verify_crc: bool) -> dict:
    archive = Path(archive)
    sequences = {split: set() for split in ANTIUAV300_SPLITS}
    required = {split: Counter() for split in ANTIUAV300_SPLITS}
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for info in members:
            parts = PurePosixPath(info.filename).parts
            for split in ANTIUAV300_SPLITS:
                if split not in parts:
                    continue
                index = parts.index(split)
                if len(parts) > index + 1 and parts[index + 1]:
                    sequence = parts[index + 1]
                    sequences[split].add(sequence)
                    if len(parts) == index + 3 and parts[index + 2] in _REQUIRED_FILES:
                        required[split][sequence] += 1
                break
        bad_member = bundle.testzip() if verify_crc else None

    return {
        "name": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "members": len(members),
        "crc_checked": verify_crc,
        "first_bad_crc_member": bad_member,
        "split_sequences": {split: len(values) for split, values in sequences.items()},
        "sequences_missing_required_members": {
            split: sorted(
                sequence for sequence in sequences[split] if required[split][sequence] != 4
            )
            for split in ANTIUAV300_SPLITS
        },
    }


def _split_has_extraction_failures(report: dict) -> bool:
    return bool(
        report["missing_sequences"]
        or report["orphan_sequences"]
        or report["missing_required_files"]
        or report["annotation_parse_errors"]
        or report["media_open_failures"]
    )


def _split_has_temporal_failures(report: dict) -> bool:
    return bool(
        _split_has_extraction_failures(report)
        or report["label_video_frame_mismatches"]
        or report["paired_video_frame_mismatches"]
        or report["paired_video_fps_mismatches"]
    )


def _split_has_annotation_failures(report: dict) -> bool:
    return bool(
        sum(
            report[key]
            for key in (
                "invalid_exist_values",
                "annotation_length_mismatches",
                "invalid_present_boxes",
                "out_of_bounds_present_boxes",
            )
        )
    )


def audit_release(
    root: Path,
    *,
    archive: Path | None = None,
    verify_archive_crc: bool = True,
    video_probe: Callable[[Path], dict] = probe_video,
) -> dict:
    """Audit Anti-UAV300 while keeping archive, extraction, and transfer gates separate."""
    root = Path(root)
    manifests = {split: load_split_manifest(root, split) for split in ANTIUAV300_SPLITS}
    manifest_sets = {split: set(values) for split, values in manifests.items()}
    overlap = {
        "train_val": sorted(manifest_sets["train"] & manifest_sets["val"]),
        "train_test": sorted(manifest_sets["train"] & manifest_sets["test"]),
        "val_test": sorted(manifest_sets["val"] & manifest_sets["test"]),
    }
    splits = {
        split: audit_split(root, split, video_probe=video_probe) for split in ANTIUAV300_SPLITS
    }

    test_dev_root = root / "test-dev"
    test_dev_sequences = (
        {path.name for path in test_dev_root.iterdir() if path.is_dir()}
        if test_dev_root.is_dir()
        else set()
    )
    test_dev_overlap = {
        split: len(test_dev_sequences & manifest_sets[split]) for split in ANTIUAV300_SPLITS
    }
    archive_report = (
        _archive_report(Path(archive), verify_crc=verify_archive_crc) if archive else None
    )
    archive_ok = archive_report is not None and (
        archive_report["first_bad_crc_member"] is None
        and archive_report["split_sequences"]
        == {split: len(manifest_sets[split]) for split in ANTIUAV300_SPLITS}
        and not any(archive_report["sequences_missing_required_members"].values())
    )
    extraction_ok = not any(_split_has_extraction_failures(report) for report in splits.values())
    annotation_ok = not any(_split_has_annotation_failures(report) for report in splits.values())
    temporal_ok = not any(_split_has_temporal_failures(report) for report in splits.values())
    train_ir = splits["train"]["modality_annotation_counts"]["infrared"]
    train_ok = (
        not _split_has_temporal_failures(splits["train"])
        and train_ir["present_frames"] > train_ir["invalid_present_boxes"]
    )
    temporal_ok = temporal_ok and all(
        splits[split]["paired_frames_compared"] > 0 for split in ANTIUAV300_SPLITS
    )

    return {
        "dataset": "Anti-UAV300",
        "root": str(root.resolve()),
        "source_archive": archive_report,
        "manifest": {
            "sequence_counts": {split: len(values) for split, values in manifest_sets.items()},
            "sequence_overlap": overlap,
        },
        "splits": splits,
        "legacy_test_dev": {
            "sequences": len(test_dev_sequences),
            "manifest_overlap": test_dev_overlap,
            "role": (
                "legacy duplicate subset; never treat as an independent validation/test split"
            ),
        },
        "gates": {
            "archive_integrity": "pass" if archive_ok else "hold",
            "extraction_completeness": "pass" if extraction_ok else "hold",
            "annotation_integrity": "pass" if annotation_ok else "hold",
            "official_manifest_sequence_disjointness": (
                "pass" if not any(overlap.values()) else "hold"
            ),
            "paired_temporal_integrity": "pass" if temporal_ok else "hold",
            "train_only_pilot_eligible": "pass" if train_ok else "hold",
            "direct_rgb_to_ir_label_transfer": "hold",
        },
        "notes": [
            "Presence mismatches can arise from the modalities' different fields of view.",
            "Frames marked present with zero-area boxes must be excluded and counted.",
            (
                "Visible and IR videos use different pixel geometries; direct box-coordinate "
                "reuse is invalid."
            ),
            (
                "A calibrated registration transform and residual acceptance threshold are "
                "still required."
            ),
        ],
    }


def load_ir_sequence_sample(
    root: Path,
    sequences: list[str],
    *,
    seed: int,
) -> tuple[list[np.ndarray], list[list[list[float]]], list[str]]:
    """Decode one deterministic annotated IR frame from each named training sequence."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Anti-UAV video sampling requires the 'detect' extra (OpenCV)"
        ) from error

    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    boxes: list[list[list[float]]] = []
    sample_ids: list[str] = []
    for sequence in sequences:
        sequence_root = Path(root) / "train" / sequence
        annotation = _load_json(sequence_root / "infrared.json")
        candidates = [
            index
            for index, (present, box) in enumerate(
                zip(annotation.get("exist", []), annotation.get("gt_rect", []), strict=False)
            )
            if (
                present == 1
                and isinstance(box, list)
                and len(box) == 4
                and float(box[2]) > 0
                and float(box[3]) > 0
            )
        ]
        if not candidates:
            raise ValueError(f"no annotated IR frame in {sequence}")
        frame_index = int(candidates[int(rng.integers(len(candidates)))])
        capture = cv2.VideoCapture(str(sequence_root / "infrared.mp4"))
        try:
            if not capture.isOpened():
                raise ValueError(f"could not open {sequence}/infrared.mp4")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise ValueError(f"could not decode {sequence} frame {frame_index}")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = [float(value) for value in annotation["gt_rect"][frame_index]]
        images.append(np.asarray(frame, dtype=np.float64))
        boxes.append([box])
        sample_ids.append(f"{sequence}:{frame_index}")
    return images, boxes, sample_ids
