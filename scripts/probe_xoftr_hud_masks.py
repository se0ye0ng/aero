"""Reviewed HUD-mask input ablation; requires human masks, never generates them."""

from __future__ import annotations

import argparse
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
from scripts.probe_detector_free_matching import VENDOR_COMMIT, WEIGHT_HASHES, read_pair
from scripts.probe_xoftr_overlay import aggregate, metrics


def merged_and_excluded(mask, header):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a 2D boolean array")
    if not 0 <= header < mask.shape[0]:
        raise ValueError("invalid header height")
    merged = mask.copy()
    merged[:header] = True
    if merged.all():
        raise ValueError("mask leaves no valid image pixels")
    excluded = cv2.dilate(merged.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    if excluded.all():
        raise ValueError("dilated mask leaves no evaluation support")
    return merged, excluded


def fill_image(image, mask, method):
    if image.ndim != 2 or image.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("grayscale image and boolean mask shapes must match")
    if mask.all():
        raise ValueError("mask leaves no valid image pixels")
    if method not in {"median", "mean"}:
        raise ValueError("fill must be median or mean")
    statistic = np.median if method == "median" else np.mean
    value = float(statistic(image[~mask]))
    output = image.copy()
    output[mask] = np.rint(value) if np.issubdtype(image.dtype, np.integer) else value
    return output, value


def retained_points(points, excluded):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("expected Nx2 points")
    h, w = excluded.shape
    keep = np.isfinite(points).all(1)
    keep &= (points >= [0, 0]).all(1) & (points <= [w - 1, h - 1]).all(1)
    indices = np.flatnonzero(keep)
    xy = np.rint(points[indices]).astype(int)
    keep[indices] &= ~excluded[xy[:, 1], xy[:, 0]]
    return keep


def filter_matches(arrays, excluded):
    a, b, _, ra, rb, _ = arrays
    forward = retained_points(a, excluded[0]) & retained_points(b, excluded[1])
    reverse = retained_points(ra, excluded[1]) & retained_points(rb, excluded[0])
    return tuple(value[forward if i < 3 else reverse] for i, value in enumerate(arrays))


def roi_support(box, mask, excluded):
    h, w = mask.shape
    x = np.arange(w)
    y = np.arange(h)
    roi = ((x >= box[0]) & (x <= box[2]))[None, :] & ((y >= box[1]) & (y <= box[3]))[:, None]
    count = int(roi.sum())
    return {
        "roi_pixel_centres": count,
        "target_occluded_fraction": float(mask[roi].mean()) if count else None,
        "target_excluded_fraction_after_dilation": float(excluded[roi].mean()) if count else None,
        "target_eligible_pixel_centres": int((roi & ~excluded).sum()),
    }


def load_masks(path, baseline, baseline_hash):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != "aero_hud_masks_v1":
        raise ValueError("requires versioned aero_hud_masks_v1 manifest")
    reviewer = manifest.get("reviewed_by")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("human reviewed_by missing")
    if manifest.get("model_blind_attestation") is not True:
        raise ValueError("human model-blind mask review attestation required")
    if manifest.get("baseline_report_sha256") != baseline_hash:
        raise ValueError("baseline report provenance mismatch")
    expected = {(p["sequence_id"], p["frame_index"]): p for p in baseline["inputs"]}
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or any(not isinstance(p, dict) for p in pairs):
        raise ValueError("mask pairs must be a list of objects")
    keys = [(p.get("sequence_id"), p.get("frame_index")) for p in pairs]
    if any(
        not isinstance(a, str) or not isinstance(b, int) or isinstance(b, bool) for a, b in keys
    ):
        raise ValueError("invalid sequence/frame ID")
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("mask panel must cover every baseline frame exactly once")
    result = {}
    for pair, key in zip(pairs, keys, strict=True):
        masks = []
        if not isinstance(pair.get("masks"), dict):
            raise ValueError("each pair requires a masks object")
        for m in ("visible", "infrared"):
            entry = pair.get("masks", {}).get(m)
            if not isinstance(entry, dict):
                raise ValueError(f"missing {m} mask")
            source = expected[key]["inputs"][m]
            if entry.get("image_sha256") != source["resized_rgb_sha256"]:
                raise ValueError("mask image provenance mismatch")
            if not isinstance(entry.get("path"), str):
                raise ValueError("mask path missing")
            mask_path = (path.parent / entry["path"]).resolve()
            if (
                not mask_path.is_relative_to(path.parent.resolve())
                or mask_path.suffix.lower() != ".png"
            ):
                raise ValueError("mask must be a relative PNG inside manifest directory")
            if file_sha256(mask_path) != entry.get("sha256"):
                raise ValueError("mask hash mismatch")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            shape = source["resized_shape"]
            if (
                mask is None
                or mask.ndim != 2
                or list(mask.shape) != shape
                or entry.get("shape") != shape
                or mask.dtype != np.uint8
            ):
                raise ValueError("mask must be uint8 grayscale with exact baseline shape")
            if not np.isin(mask, [0, 255]).all():
                raise ValueError("mask must be binary 0/255; no antialiasing")
            masks.append(mask == 255)
        result[key] = masks
    return manifest, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("experiments/detector_free_train16_01/report.json"),
    )
    parser.add_argument(
        "--root", type=Path, default=antiuav300_root()
    )
    parser.add_argument("--external-root", type=Path, default=Path("experiments/external"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite HUD ablation")
    baseline = json.loads(args.baseline_report.read_text())
    if (
        baseline.get("evaluated_split") != "train"
        or baseline.get("validation_or_test_access") != "none"
    ):
        raise ValueError("train-only baseline required")
    manifest, masks = load_masks(args.mask_manifest, baseline, file_sha256(args.baseline_report))
    cases = [r for r in baseline["rows"] if r["model"] == "xoftr"]
    keys = [(r["sequence_id"], r["frame_index"]) for r in cases]
    if len(cases) != 16 or len(set(keys)) != 16 or set(keys) != set(masks):
        raise ValueError("requires exact original 16-frame baseline")
    if baseline["protocol"]["long_side"] != 640:
        raise ValueError("requires original 640px baseline")
    for pair_masks in masks.values():
        for mask, shape, top in zip(pair_masks, [(360, 640), (512, 640)], [72, 104], strict=True):
            if mask.shape != shape:
                raise ValueError("requires exact original image/mask geometry")
            merged_and_excluded(mask, top)
    if file_sha256(args.root / "label_new/train.json") != baseline["split_manifest_sha256"]:
        raise ValueError("train split changed")
    project = Path(__file__).resolve().parents[1]
    for source, digest in baseline["sources"].items():
        if file_sha256(project / source) != digest:
            raise ValueError(f"baseline source changed: {source}")
    vendor = args.external_root / "XoFTR"
    commit = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True)
    weights = args.external_root / "weights_xoftr_640.ckpt"
    if (
        commit != VENDOR_COMMIT
        or dirty.strip()
        or file_sha256(weights) != WEIGHT_HASHES["xoftr"]
        or baseline["weights_sha256"]["xoftr"] != WEIGHT_HASHES["xoftr"]
    ):
        raise ValueError("vendor/weights provenance mismatch")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    model = load_matcher("xoftr", weights, vendor.resolve(), args.device)
    inputs = {(r["sequence_id"], r["frame_index"]): r for r in baseline["inputs"]}
    args.output_dir.mkdir(parents=True)
    rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for index, case in enumerate(cases):
            key = (case["sequence_id"], case["frame_index"])
            images, boxes, provenance = read_pair(args.root, key[0], 640)
            if provenance != inputs[key]:
                raise ValueError("decoded image provenance changed")
            grays = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]
            if [g.shape for g in grays] != [(360, 640), (512, 640)]:
                raise ValueError("unexpected image geometry")
            merged, excluded = zip(
                *(
                    merged_and_excluded(mask, top)
                    for mask, top in zip(masks[key], [72, 104], strict=True)
                ),
                strict=True,
            )
            support = [
                roi_support(box, mask, exclusion)
                for box, mask, exclusion in zip(boxes, merged, excluded, strict=True)
            ]
            match_path = (args.baseline_report.parent / case["matches_file"]).resolve()
            if not match_path.is_relative_to(args.baseline_report.parent.resolve()):
                raise ValueError("baseline matches path outside report directory")
            if file_sha256(match_path) != case["matches_sha256"]:
                raise ValueError("baseline matches hash mismatch")
            with np.load(match_path, allow_pickle=False) as saved:
                raw = tuple(
                    saved[k]
                    for k in (
                        "points0",
                        "points1",
                        "confidence",
                        "reverse0",
                        "reverse1",
                        "reverse_confidence",
                    )
                )
            for condition in ("raw_postfilter", "median_fill", "mean_fill"):
                flags, fill_values = {}, []
                if condition == "raw_postfilter":
                    arrays = raw
                else:
                    filled = [
                        fill_image(g, mask, condition.split("_")[0])
                        for g, mask in zip(grays, merged, strict=True)
                    ]
                    fill_values = [v for _, v in filled]
                    pair_images = [g for g, _ in filled]
                    a, b, c, f = infer(model, "xoftr", *pair_images, args.device)
                    ra, rb, rc, rf = infer(model, "xoftr", *pair_images[::-1], args.device)
                    arrays = (a, b, c, ra, rb, rc)
                    flags = {"forward": f, "reverse": rf}
                a, b, c, ra, rb, rc = filter_matches(arrays, excluded)
                stats, _ = metrics(a, b, ra, rb, boxes)
                destination = args.output_dir / f"{index:03d}_{condition}.npz"
                np.savez_compressed(
                    destination,
                    points0=a,
                    points1=b,
                    confidence=c,
                    reverse0=ra,
                    reverse1=rb,
                    reverse_confidence=rc,
                )
                rows.append(
                    {
                        "sequence_id": key[0],
                        "frame_index": key[1],
                        "condition": condition,
                        "metrics": stats,
                        "roi_support": support,
                        "target_fully_retained": all(
                            s["target_excluded_fraction_after_dilation"] == 0 for s in support
                        ),
                        "boxes_xyxy": [b.tolist() for b in boxes],
                        "flags": flags,
                        "fill_values": fill_values,
                        "matches_file": destination.name,
                        "matches_sha256": file_sha256(destination),
                    }
                )
            print(f"{index + 1}/16 reviewed HUD-mask comparison complete", flush=True)
    report = {
        "kind": "reviewed_hud_mask_input_ablation",
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "device": args.device,
        "elapsed_seconds": time.monotonic() - started,
        "baseline_sha256": file_sha256(args.baseline_report),
        "mask_manifest_sha256": file_sha256(args.mask_manifest),
        "mask_manifest": manifest,
        "source_sha256": file_sha256(__file__),
        "rows": rows,
        "summary": aggregate(rows),
        "protocol": {
            "header_rows": [72, 104],
            "exclusion_dilation_chebyshev_px": 8,
            "same_evaluation_support_all_conditions": True,
            "image_resize": False,
            "fill_statistics": "unmasked pixels only, integer fills rounded to nearest",
            "mask_generation": "external human reviewed, no GT box-driven mask generation",
            "inpainting": False,
            "pixel_ground_truth_available": False,
        },
        "generator_training_eligible": "hold",
    }
    with (args.output_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
