#!/usr/bin/env python3
"""Input header-crop ablation; correspondence proxies are NOT qualification."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.detector_free import infer, inside_box, load_matcher, pair_metrics
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_detector_free_screen import hull_fraction, transfer_iou
from scripts.probe_detector_free_matching import (
    VENDOR_COMMIT,
    WEIGHT_HASHES,
    read_pair,
    visual_review,
)


def header_rows(height):
    """Top20%, rounded upward to eight rows; independent of boxes/matches."""
    return int(np.ceil(height * 0.2 / 8) * 8)


def restore(points, top):
    return np.asarray(points) + np.array([0, top])


def retained(points0, points1, tops):
    return (points0[:, 1] >= tops[0]) & (points1[:, 1] >= tops[1])


def metrics(a, b, ra, rb, boxes):
    forward, reciprocal = pair_metrics(a, b, ra, rb, *boxes)
    reverse, _ = pair_metrics(ra, rb, a, b, *boxes[::-1])
    both = inside_box(a, boxes[0]) & inside_box(b, boxes[1]) & reciprocal
    return {
        "rgb_to_ir": forward,
        "ir_to_rgb": reverse,
        "unique_reciprocal_both_box_pairs": len(
            np.unique(np.round(np.c_[a[both], b[both]]), axis=0)
        ),
        "rgb_hull_fraction": hull_fraction(a[both], boxes[0]),
        "ir_hull_fraction": hull_fraction(b[both], boxes[1]),
        "rgb_to_ir_homography_box_iou_proxy": transfer_iou(
            forward["homography_consistency_not_accuracy"].get("matrix"), *boxes
        ),
    }, reciprocal


def aggregate(rows):
    result = {}
    for condition in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == condition]
        common = [r for r in group if r["target_fully_retained"]]
        for suffix, cases in (("all", group), ("common_target", common)):
            vals = [r["metrics"] for r in cases]
            ious = [v["rgb_to_ir_homography_box_iou_proxy"] for v in vals]
            finite = [v for v in ious if v is not None]
            result[f"{condition}/{suffix}"] = {
                "pairs": len(vals),
                "median_matches": float(np.median([v["rgb_to_ir"]["matches"] for v in vals]))
                if vals
                else None,
                "pairs_with_both_box_matches": sum(
                    v["rgb_to_ir"]["both_box_matches"] > 0 for v in vals
                ),
                "pairs_with_four_unique_reciprocal_both_box": sum(
                    v["unique_reciprocal_both_box_pairs"] >= 4 for v in vals
                ),
                "pairs_also_covering_ten_percent_both_boxes": sum(
                    v["unique_reciprocal_both_box_pairs"] >= 4
                    and min(v["rgb_hull_fraction"], v["ir_hull_fraction"]) >= 0.1
                    for v in vals
                ),
                "homography_box_iou_proxy_available": len(finite),
                "homography_box_iou_proxy_mean": float(np.mean(finite)) if finite else None,
                "homography_box_iou_proxy_ge_0_6": sum(v >= 0.6 for v in finite),
            }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300"))
    p.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("experiments/detector_free_train16_01/report.json"),
    )
    p.add_argument("--external-root", type=Path, default=Path("experiments/external"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("choose a fresh output directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    baseline = json.loads(args.baseline_report.read_text())
    project = Path(__file__).resolve().parents[1]
    if baseline["evaluated_split"] != "train" or baseline["validation_or_test_access"] != "none":
        raise ValueError("requires train-only baseline")
    if file_sha256(args.root / "label_new/train.json") != baseline["split_manifest_sha256"]:
        raise ValueError("split changed")
    for path, digest in baseline["sources"].items():
        if file_sha256(project / path) != digest:
            raise ValueError(f"baseline source changed: {path}")
    vendor = args.external_root / "XoFTR"
    commit = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True)
    weights = args.external_root / "weights_xoftr_640.ckpt"
    if commit != VENDOR_COMMIT or dirty.strip() or file_sha256(weights) != WEIGHT_HASHES["xoftr"]:
        raise ValueError("vendor/weights not pinned")
    if baseline["weights_sha256"]["xoftr"] != WEIGHT_HASHES["xoftr"]:
        raise ValueError("baseline weights differ")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    model = load_matcher("xoftr", weights, vendor.resolve(), args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {r["sequence_id"]: r for r in baseline["inputs"]}
    cases = [r for r in baseline["rows"] if r["model"] == "xoftr"]
    rows, controls = [], []
    start = time.monotonic()
    first_gray = None
    with torch.inference_mode():
        for index, case in enumerate(cases):
            images, boxes, provenance = read_pair(
                args.root, case["sequence_id"], baseline["protocol"]["long_side"]
            )
            if provenance != inputs[case["sequence_id"]]:
                raise ValueError("decoded pair/provenance changed")
            grays = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images]
            tops = [header_rows(g.shape[0]) for g in grays]
            cropped = [g[t:] for g, t in zip(grays, tops, strict=True)]
            match_path = args.baseline_report.parent / case["matches_file"]
            if file_sha256(match_path) != case["matches_sha256"]:
                raise ValueError("baseline matches changed")
            with np.load(match_path) as saved:
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
            a, b, c, flags = infer(model, "xoftr", *cropped, args.device)
            ra, rb, rc, rflags = infer(model, "xoftr", *cropped[::-1], args.device)
            ablated = (
                restore(a, tops[0]),
                restore(b, tops[1]),
                c,
                restore(ra, tops[1]),
                restore(rb, tops[0]),
                rc,
            )
            keep, rkeep = retained(raw[0], raw[1], tops), retained(raw[3], raw[4], tops[::-1])
            filtered = tuple(value[keep if j < 3 else rkeep] for j, value in enumerate(raw))
            for condition, arrays in (
                ("raw", raw),
                ("raw_postfilter", filtered),
                ("input_header_crop", ablated),
            ):
                a, b, c, ra, rb, rc = arrays
                stats, reciprocal = metrics(a, b, ra, rb, boxes)
                path = args.output_dir / f"{index:03d}_{condition}.npz"
                np.savez_compressed(
                    path,
                    points0=a,
                    points1=b,
                    confidence=c,
                    reverse0=ra,
                    reverse1=rb,
                    reverse_confidence=rc,
                )
                row = {
                    "sequence_id": case["sequence_id"],
                    "frame_index": case["frame_index"],
                    "condition": condition,
                    "header_rows": tops,
                    "boxes_xyxy": [b.tolist() for b in boxes],
                    "target_fully_retained": all(
                        box[1] >= top for box, top in zip(boxes, tops, strict=True)
                    ),
                    "metrics": stats,
                    "matches_file": path.name,
                    "matches_sha256": file_sha256(path),
                }
                if condition == "input_header_crop":
                    row["flags"] = {"forward": flags, "reverse": rflags}
                rows.append(row)
                if condition != "raw":
                    visual_review(
                        args.output_dir / f"{index:03d}_{condition}.png",
                        images,
                        boxes,
                        a,
                        b,
                        c,
                        reciprocal,
                        f"{condition}: {case['sequence_id']}; repeatability NOT accuracy",
                    )
            if first_gray is None:
                first_gray = cropped[0].copy()
            if index in (0, len(cases) - 1):
                gray = cropped[0]
                shifted = cv2.warpAffine(
                    gray, np.float32([[1, 0, 8], [0, 1, -8]]), (gray.shape[1], gray.shape[0])
                )
                x, y, _, f = infer(model, "xoftr", gray, shifted, args.device)
                error = np.linalg.norm(y - x - [8, -8], axis=1)
                controls.append(
                    {
                        "kind": "known_shift",
                        "sequence_id": case["sequence_id"],
                        "matches": len(x),
                        "median_error_px": float(np.median(error)) if len(x) else None,
                        "fraction_within_3px": float((error <= 3).mean()) if len(x) else None,
                        "flags": f,
                    }
                )
            print(f"{index + 1}/{len(cases)} header crop evaluated", flush=True)
        x, y, confidence, flags = infer(model, "xoftr", first_gray, cropped[1], args.device)
        path = args.output_dir / "negative_header_crop.npz"
        np.savez_compressed(path, points0=x, points1=y, confidence=confidence)
        controls.append(
            {
                "kind": "unrelated_sequence",
                "rgb_sequence": cases[0]["sequence_id"],
                "ir_sequence": cases[-1]["sequence_id"],
                "matches": len(x),
                "flags": flags,
                "matches_file": path.name,
                "matches_sha256": file_sha256(path),
            }
        )
    sources = [
        Path(__file__),
        project / "src/aero_ir/registration/detector_free.py",
        project / "scripts/analyze_detector_free_screen.py",
    ]
    report = {
        "experiment": "xoftr_input_header_ablation_v1",
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "device": args.device,
        "torch_version": torch.__version__,
        "baseline_report": str(args.baseline_report),
        "baseline_sha256": file_sha256(args.baseline_report),
        "baseline_matches_reused_with_hash_validation": True,
        "weights_sha256": file_sha256(weights),
        "vendor_commit": commit,
        "elapsed_seconds": time.monotonic() - start,
        "inputs": list(inputs.values()),
        "rows": rows,
        "controls": controls,
        "summary": aggregate(rows),
        "protocol": {
            "crop": "top20% ceil8; no rescale; restore y offset",
            "primary_comparison": (
                "raw_postfilter versus input_header_crop on identical retained domain"
            ),
            "central_reticles_removed": False,
            "pixel_gt_available": False,
            "ten_percent_hull": "descriptive, not qualification threshold",
        },
        "sources": {str(s.relative_to(project)): file_sha256(s) for s in sources},
        "generator_training_eligible": "hold",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
