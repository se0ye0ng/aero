"""Human-free, box-assisted train diagnostic; automatic matches are NOT pixel GT.

Native crops preserve information lost by the previous full-RGB 640px resize.
Identical automatic HUD exclusion support is used for raw and inpainted inputs.
Shuffled sequence pairs retain UAVs in both crops: they are hard negatives for
frame-specific correspondence, not guaranteed negatives for semantic similarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.utils.manifest import file_sha256
from aero_ir.utils.paths import antiuav300_root
from scripts.audit_antiuav300_dense_registration import _candidate_pairs, _read_at
from scripts.probe_detector_free_matching import VENDOR_COMMIT, WEIGHT_HASHES
from scripts.probe_xoftr_hud_masks import filter_matches
from scripts.probe_xoftr_overlay import metrics


def auto_hud(gray):
    """Fixed header plus thin axis-aligned extreme-intensity line candidates.

    This is a heuristic, not a true overlay mask; real scene edges may be removed.
    No boxes, correspondences, or human masks enter this detector.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("expected uint8 grayscale")
    h, w = gray.shape
    central = np.zeros_like(gray)
    for extreme in ((gray <= 10), (gray >= 245)):
        for horizontal in (True, False):
            length = max(16, round((w if horizontal else h) * 0.07))
            kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
            opened = cv2.morphologyEx(extreme.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
            for i in range(1, count):
                x, y, bw, bh, _ = stats[i]
                thickness, extent = (bh, bw) if horizontal else (bw, bh)
                if (
                    thickness <= 4
                    and extent >= length
                    and 0.18 * w <= x + bw / 2 <= 0.82 * w
                    and 0.18 * h <= y + bh / 2 <= 0.82 * h
                ):
                    central[labels == i] = 1
    mask = cv2.dilate(central, np.ones((3, 3), np.uint8)).astype(bool)
    mask[: int(np.ceil(0.2 * h))] = True
    return mask


def crop_region(image, box, side=256):
    """Square 3x-box context, clipped at image boundaries, half-pixel coordinates."""
    box = np.asarray(box, float)
    if (
        box.shape != (4,)
        or not np.isfinite(box).all()
        or np.any(box[2:] <= box[:2])
        or side < 32
        or side % 8
    ):
        raise ValueError("invalid box or crop size")
    h, w = image.shape[:2]
    centre = (box[:2] + box[2:]) / 2
    radius = max(float((box[2:] - box[:2]).max()) * 1.5, 24)
    low = np.maximum(np.floor(centre - radius).astype(int), [0, 0])
    high = np.minimum(np.ceil(centre + radius).astype(int), [w, h])
    if np.any(high <= low):
        raise ValueError("crop outside image")
    x0, y0 = low
    x1, y1 = high
    region = cv2.resize(image[y0:y1, x0:x1], (side, side), interpolation=cv2.INTER_LINEAR)
    scale = np.array([side / (x1 - x0), side / (y1 - y0)])
    cropped_box = (box.reshape(2, 2) - low + 0.5) * scale - 0.5
    metadata = {
        "bounds_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "scale_xy": scale.tolist(),
        "output_side": side,
    }
    return region, cropped_box.ravel(), metadata


def restore_native(points, metadata):
    return (
        (np.asarray(points) + 0.5) / metadata["scale_xy"]
        - 0.5
        + np.array(metadata["bounds_xyxy"][:2])
    )


def prepare(gray, box, side):
    mask = auto_hud(gray)
    filled = cv2.inpaint(gray, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    raw, cb, meta = crop_region(gray, box, side)
    clean, _, _ = crop_region(filled, box, side)
    x0, y0, x1, y1 = meta["bounds_xyxy"]
    # Any bilinear contribution from a masked pixel is excluded, plus an 8px
    # model-grid guard band. Inpainted pixels never count as correspondences.
    excluded = cv2.resize(mask[y0:y1, x0:x1].astype(np.float32), (side, side)) > 0
    excluded = cv2.dilate(excluded.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    yy, xx = np.mgrid[:side, :side]
    roi = (xx >= cb[0]) & (xx <= cb[2]) & (yy >= cb[1]) & (yy <= cb[3])
    meta["target_excluded_fraction"] = float(excluded[roi].mean()) if roi.any() else None
    return {"raw": raw, "inpaint": clean, "excluded": excluded, "box": cb, "meta": meta}


def load_native(root, record):
    folder = root / "train" / record["sequence_id"]
    indices, rgb_boxes, ir_boxes = _candidate_pairs(folder)
    pos = len(indices) // 2
    if indices[pos] != record["frame_index"]:
        raise ValueError("baseline midpoint changed")
    result = []
    for modality, box in zip(("visible", "infrared"), (rgb_boxes[pos], ir_boxes[pos]), strict=True):
        old = record["inputs"][modality]
        if file_sha256(folder / f"{modality}.json") != old["annotation_sha256"]:
            raise ValueError("annotation changed")
        path = folder / f"{modality}.mp4"
        cap = cv2.VideoCapture(str(path))
        try:
            bgr = _read_at(cap, indices[pos], path)
        finally:
            cap.release()
        if hashlib.sha256(bgr.tobytes()).hexdigest() != old["decoded_native_sha256"]:
            raise ValueError("decoded native frame changed")
        h, w = bgr.shape[:2]
        cx, cy, bw, bh = box
        xyxy = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        xyxy *= [w, h, w, h]
        result.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), xyxy))
    return result


def run_matches(model, left, right, condition, device):
    a, b, c, f = infer(model, "xoftr", left[condition], right[condition], device)
    ra, rb, rc, rf = infer(model, "xoftr", right[condition], left[condition], device)
    arrays = filter_matches((a, b, c, ra, rb, rc), [left["excluded"], right["excluded"]])
    a, b, c, ra, rb, rc = arrays
    stats, _ = metrics(a, b, ra, rb, [left["box"], right["box"]])
    # Box overlap here would be circular evidence: boxes already centre the crops.
    stats.pop("rgb_to_ir_homography_box_iou_proxy")
    return arrays, stats, {"forward": f, "reverse": rf}


def supported(stats):
    return (
        stats["unique_reciprocal_both_box_pairs"] >= 4
        and min(stats["rgb_hull_fraction"], stats["ir_hull_fraction"]) >= 0.1
    )


def aggregate(rows):
    result = {}
    for condition in ("raw", "inpaint"):
        for kind in ("paired", "shuffled"):
            group = [r for r in rows if r["condition"] == condition and r["kind"] == kind]
            stats = [r["metrics"] for r in group]
            result[f"{condition}/{kind}"] = {
                "pairs": len(group),
                "any_both_box": sum(s["rgb_to_ir"]["both_box_matches"] > 0 for s in stats),
                "four_unique_reciprocal_both_box": sum(
                    s["unique_reciprocal_both_box_pairs"] >= 4 for s in stats
                ),
                "also_ten_percent_hull_both_boxes": sum(supported(s) for s in stats),
                "median_unique_reciprocal_both_box": float(
                    np.median([s["unique_reciprocal_both_box_pairs"] for s in stats])
                )
                if stats
                else None,
            }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=antiuav300_root())
    p.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("experiments/detector_free_train16_01/report.json"),
    )
    p.add_argument("--external-root", type=Path, default=Path("experiments/external"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--side", type=int, default=256)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("choose fresh output directory")
    if args.side < 32 or args.side % 8:
        p.error("side must be >=32 and divisible by 8")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    baseline = json.loads(args.baseline_report.read_text())
    if baseline["evaluated_split"] != "train" or baseline["validation_or_test_access"] != "none":
        raise ValueError("train-only input required")
    if file_sha256(args.root / "label_new/train.json") != baseline["split_manifest_sha256"]:
        raise ValueError("train split changed")
    records = baseline["inputs"]
    if len(records) < 2 or len({r["sequence_id"] for r in records}) != len(records):
        raise ValueError("distinct sequences required for shuffled controls")
    vendor, weights = args.external_root / "XoFTR", args.external_root / "weights_xoftr_640.ckpt"
    revision = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True)
    if revision != VENDOR_COMMIT or dirty.strip() or file_sha256(weights) != WEIGHT_HASHES["xoftr"]:
        raise ValueError("pinned model provenance mismatch")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    model = load_matcher("xoftr", weights, vendor.resolve(), args.device)
    started = time.monotonic()
    prepared = [[prepare(g, b, args.side) for g, b in load_native(args.root, r)] for r in records]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, controls, artifacts = [], [], {}
    for i, pair in enumerate(prepared):
        for m, side in zip(("rgb", "ir"), pair, strict=True):
            for field in ("raw", "inpaint", "excluded"):
                path = args.output_dir / f"{i:03d}_{m}_{field}.png"
                value = side[field].astype(np.uint8) * 255 if field == "excluded" else side[field]
                if not cv2.imwrite(str(path), value):
                    raise OSError(f"could not save {path}")
                artifacts[path.name] = file_sha256(path)
    with torch.inference_mode():
        for i, (left, right) in enumerate(prepared):
            for condition in ("raw", "inpaint"):
                for kind, j in (("paired", i), ("shuffled", (i + 1) % len(prepared))):
                    target = prepared[j][1]
                    arrays, stats, flags = run_matches(model, left, target, condition, args.device)
                    path = args.output_dir / f"{i:03d}_{condition}_{kind}.npz"
                    names = (
                        "points0",
                        "points1",
                        "confidence",
                        "reverse0",
                        "reverse1",
                        "reverse_confidence",
                    )
                    output = dict(zip(names, arrays, strict=True))
                    output["native_points0"] = restore_native(arrays[0], left["meta"])
                    output["native_points1"] = restore_native(arrays[1], target["meta"])
                    np.savez_compressed(path, **output)
                    artifacts[path.name] = file_sha256(path)
                    rows.append(
                        {
                            "sequence_id": records[i]["sequence_id"],
                            "target_sequence_id": records[j]["sequence_id"],
                            "condition": condition,
                            "kind": kind,
                            "metrics": stats,
                            "crop_metadata": [left["meta"], target["meta"]],
                            "flags": flags,
                            "matches_file": path.name,
                        }
                    )
            if i in (0, len(prepared) - 1):
                for modality, side in (("rgb", left), ("ir", right)):
                    g = side["raw"]
                    shifted = cv2.warpAffine(g, np.float32([[1, 0, 8], [0, 1, -8]]), g.shape[::-1])
                    a, b, _, flags = infer(model, "xoftr", g, shifted, args.device)
                    valid = ((a >= 16) & (a < args.side - 16)).all(1)
                    valid &= ((b >= 16) & (b < args.side - 16)).all(1)
                    errors = np.linalg.norm(b[valid] - a[valid] - [8, -8], axis=1)
                    controls.append(
                        {
                            "sequence_id": records[i]["sequence_id"],
                            "modality": modality,
                            "known_shift_model_pixels": [8, -8],
                            "matches": len(errors),
                            "fraction_within_3px": float((errors <= 3).mean())
                            if len(errors)
                            else None,
                            "flags": flags,
                        }
                    )
            print(f"{i + 1}/{len(prepared)} paired + shuffled ROI checks", flush=True)
    project = Path(__file__).resolve().parents[1]
    sources = (
        "scripts/probe_registration_auto_roi.py",
        "scripts/probe_xoftr_overlay.py",
        "scripts/probe_xoftr_hud_masks.py",
        "scripts/probe_detector_free_matching.py",
        "scripts/analyze_detector_free_screen.py",
        "scripts/audit_antiuav300_dense_registration.py",
        "src/aero_ir/registration/detector_free.py",
    )
    report = {
        "experiment": "automatic_box_assisted_native_roi_v1",
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "torch_version": torch.__version__,
        "baseline_report_sha256": file_sha256(args.baseline_report),
        "inputs": records,
        "weights_sha256": file_sha256(weights),
        "vendor_commit": revision,
        "protocol": {
            "human_input": False,
            "existing_train_boxes_used_for_cropping": True,
            "is_annotation_free": False,
            "box_iou_is_not_independent_evidence": True,
            "crop_side": args.side,
            "context": "square 3x max box side, minimum48 nativepx",
            "negative": "cyclic next sequence; semantic similarity possible, not same frame",
            "coordinate_convention": "pixel_centres_half_pixel_resize",
            "hud": "heuristic header20percent plus extreme-intensity thin axis-aligned lines",
            "common_exclusion_guard_model_px": 8,
            "scope": "target ROI hypothesis screen, not full-image registration",
            "no_new_model_training": True,
        },
        "summary": aggregate(rows),
        "rows": rows,
        "synthetic_controls": controls,
        "artifacts_sha256": artifacts,
        "sources": {s: file_sha256(project / s) for s in sources},
        "qualification": "not_assessed_no_independent_pixel_gt",
    }
    with (args.output_dir / "report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
