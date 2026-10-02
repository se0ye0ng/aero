#!/usr/bin/env python3
"""Paired XoFTR/LoFTR train-only correspondence screen at native-derived 640px.

No training, GT-driven match selection, dense eligibility, or test/val access.
Box routing and model consistency are proxies, never pixel correspondence GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

from aero_ir.data.antiuav import load_split_manifest
from aero_ir.registration.detector_free import (
    header_diagnostic,
    infer,
    inside_box,
    load_matcher,
    pair_metrics,
)
from aero_ir.utils.manifest import file_sha256
from scripts.audit_antiuav300_dense_registration import _candidate_pairs, _read_at

VENDOR_COMMIT = "e0fbea431b30be9742effbf5577c90aa8eb938f9"
WEIGHT_HASHES = {
    "loftr": "21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec",
    "xoftr": "6e7f24e553f76746ef49061bf44c7d11252e56057470d1421bcd4e6506af3a67",
}


def read_pair(root, sequence, long_side):
    folder = root / "train" / sequence
    indices, boxes0, boxes1 = _candidate_pairs(folder)
    position = len(indices) // 2
    frame = indices[position]
    images, boxes, provenance = [], [], {}
    for name, box in zip(
        ("visible", "infrared"), (boxes0[position], boxes1[position]), strict=True
    ):
        path = folder / f"{name}.mp4"
        capture = cv2.VideoCapture(str(path))
        try:
            image = _read_at(capture, frame, path)
        finally:
            capture.release()
        h, w = image.shape[:2]
        expected = (1080, 1920) if name == "visible" else (512, 640)
        if (h, w) != expected:
            raise ValueError(f"unexpected native shape: {name} {(h, w)}")
        ratio = long_side / max(h, w)
        new_w, new_h = int(w * ratio) // 8 * 8, int(h * ratio) // 8 * 8
        resized = cv2.cvtColor(cv2.resize(image, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        cx, cy, bw, bh = box
        boxes.append(
            np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            * np.array([new_w, new_h, new_w, new_h])
        )
        images.append(resized)
        provenance[name] = {
            "annotation_sha256": file_sha256(folder / f"{name}.json"),
            "native_shape": [h, w],
            "resized_shape": [new_h, new_w],
            "decoded_native_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "resized_rgb_sha256": hashlib.sha256(resized.tobytes()).hexdigest(),
        }
    return (
        images,
        boxes,
        {
            "sequence_id": sequence,
            "frame_index": frame,
            "usable_position": position,
            "inputs": provenance,
        },
    )


def visual_review(path, images, boxes, p0, p1, confidence, reciprocal, title):
    """Full views and target crops; colored lines indicate repeatability, NOT truth."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row in range(2):
        for col, (image, box) in enumerate(zip(images, boxes, strict=True)):
            ax = axes[row, col]
            ax.imshow(image)
            ax.add_patch(
                Rectangle(box[:2], *(box[2:] - box[:2]), fill=False, edgecolor="red", lw=1.5)
            )
            if row == 1:
                centre = (box[:2] + box[2:]) / 2
                radius = np.maximum((box[2:] - box[:2]) * 1.5, 48)
                low, high = (
                    np.maximum(centre - radius, 0),
                    np.minimum(centre + radius, np.array([image.shape[1], image.shape[0]]) - 1),
                )
                ax.set_xlim(low[0], high[0])
                ax.set_ylim(high[1], low[1])
            ax.set_title(("RGB" if col == 0 else "IR") + (" target crop" if row else " full view"))
        roi_either = inside_box(p0, boxes[0]) | inside_box(p1, boxes[1])
        choices = np.flatnonzero(roi_either) if row else np.arange(len(p0))
        choices = choices[np.argsort(-confidence[choices], kind="stable")[: 40 if row == 0 else 20]]
        for i in choices:
            color = "lime" if reciprocal[i] else "orange"
            for col, points in enumerate((p0, p1)):
                axes[row, col].plot(*points[i], ".", color=color, markersize=4)
                if row:
                    axes[row, col].text(*points[i], str(i), fontsize=6, color=color, clip_on=True)
            if not row:
                fig.add_artist(
                    ConnectionPatch(
                        p0[i],
                        p1[i],
                        "data",
                        "data",
                        axesA=axes[row, 0],
                        axesB=axes[row, 1],
                        color=color,
                        alpha=0.4,
                        lw=0.5,
                    )
                )
    fig.suptitle(
        title + "\nGreen=reciprocal within 2px, orange=not; neither is pixel GT. Red=GT box."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary(rows):
    output = {}
    for name in ("xoftr", "loftr"):
        cases = [r for r in rows if r["model"] == name]
        for direction in ("rgb_to_ir", "ir_to_rgb"):
            stats = [r[direction] for r in cases]
            h = [r["homography_consistency_not_accuracy"] for r in stats]
            supported = [r for r in h if r["status"] == "fit"]
            output[f"{name}/{direction}"] = {
                "pairs": len(cases),
                "median_matches": float(np.median([r["matches"] for r in stats])),
                "median_non_header_matches": float(
                    np.median([r["header_screen"]["non_header_matches"] for r in stats])
                ),
                "total_header_either_matches": sum(
                    r["header_screen"]["header_either_matches"] for r in stats
                ),
                "total_matches": sum(r["matches"] for r in stats),
                "pairs_no_matches": sum(r["matches"] == 0 for r in stats),
                "pairs_with_any_both_box_match": sum(r["both_box_matches"] > 0 for r in stats),
                "pairs_with_four_both_box_reciprocal_matches": sum(
                    r["both_box_reciprocal_matches"] >= 4 for r in stats
                ),
                "median_both_box_matches": float(np.median([r["both_box_matches"] for r in stats])),
                "total_source_box_matches": sum(r["source_box_matches"] for r in stats),
                "total_both_box_matches": sum(r["both_box_matches"] for r in stats),
                "homographies_fit": len(supported),
                "median_check_fraction_within_3px": float(
                    np.median([r["check_fraction_within_3px"] for r in supported])
                )
                if supported
                else None,
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300")
    )
    parser.add_argument("--external-root", type=Path, default=Path("experiments/external"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.long_side < 128 or args.long_side % 8:
        parser.error("long side must be >=128 and divisible by 8")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    vendor = args.external_root / "XoFTR"
    commit = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True)
    if commit != VENDOR_COMMIT or dirty.strip():
        raise ValueError("XoFTR checkout must be clean and pinned")
    weights = {
        "xoftr": args.external_root / "weights_xoftr_640.ckpt",
        "loftr": args.external_root / "loftr_outdoor.ckpt",
    }
    hashes = {k: file_sha256(p) for k, p in weights.items()}
    if hashes != WEIGHT_HASHES:
        raise ValueError(f"unexpected model weight hashes: {hashes}")
    sequences = sorted(load_split_manifest(args.root, "train"))
    if not 1 <= args.sequences <= len(sequences):
        parser.error("invalid sequence count")
    chosen = [sequences[i] for i in np.linspace(0, len(sequences) - 1, args.sequences, dtype=int)]
    models = {
        name: load_matcher(name, path, vendor.resolve(), args.device)
        for name, path in weights.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, controls, provenance, negative_controls = [], [], [], []
    first_gray = None
    start = time.monotonic()
    with torch.inference_mode():
        for index, sequence in enumerate(chosen):
            images, boxes, inputs = read_pair(args.root, sequence, args.long_side)
            provenance.append(inputs)
            grays = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
            if first_gray is None:
                first_gray = grays[0].copy()
            for name, model in models.items():
                tick = time.monotonic()
                a, b, confidence, flags = infer(model, name, *grays, args.device)
                ra, rb, reverse_confidence, reverse_flags = infer(
                    model, name, *grays[::-1], args.device
                )
                forward, reciprocal = pair_metrics(a, b, ra, rb, *boxes)
                reverse, _ = pair_metrics(ra, rb, a, b, *boxes[::-1])
                forward["header_screen"] = header_diagnostic(
                    a, b, grays[0].shape[0], grays[1].shape[0]
                )
                reverse["header_screen"] = header_diagnostic(
                    ra, rb, grays[1].shape[0], grays[0].shape[0]
                )
                match_file = args.output_dir / f"{index:03d}_{name}_matches.npz"
                np.savez_compressed(
                    match_file,
                    points0=a,
                    points1=b,
                    confidence=confidence,
                    reverse0=ra,
                    reverse1=rb,
                    reverse_confidence=reverse_confidence,
                )
                row = {
                    "model": name,
                    "sequence_id": sequence,
                    "frame_index": inputs["frame_index"],
                    "rgb_to_ir": forward,
                    "ir_to_rgb": reverse,
                    "forward_flags": flags,
                    "reverse_flags": reverse_flags,
                    "seconds": time.monotonic() - tick,
                    "matches_file": match_file.name,
                    "matches_sha256": file_sha256(match_file),
                }
                rows.append(row)
                visual_review(
                    args.output_dir / f"{index:03d}_{name}_review.png",
                    images,
                    boxes,
                    a,
                    b,
                    confidence,
                    reciprocal,
                    f"{name}: {sequence} frame {inputs['frame_index']}, matches={len(a)}",
                )
                # Known geometry on first and last selected sequence; RGB only,
                # no claim that this models all cross-modal appearance changes.
                if index in (0, len(chosen) - 1):
                    gray = grays[0]
                    shifted = cv2.warpAffine(
                        gray, np.float32([[1, 0, 8], [0, 1, -8]]), (gray.shape[1], gray.shape[0])
                    )
                    c0, c1, _, sf = infer(model, name, gray, shifted, args.device)
                    errors = np.linalg.norm(c1 - c0 - [8, -8], axis=1)
                    controls.append(
                        {
                            "model": name,
                            "sequence_id": sequence,
                            "matches": len(c0),
                            "flags": sf,
                            "fraction_within_3px": float((errors <= 3).mean()) if len(c0) else None,
                            "median_error_px": float(np.median(errors)) if len(c0) else None,
                        }
                    )
                print(
                    f"{index + 1}/{len(chosen)} {name}: {len(a)} matches, "
                    f"{forward['both_box_matches']} both-box, "
                    f"{forward['both_box_reciprocal_matches']} both-box reciprocal",
                    flush=True,
                )
                if index == len(chosen) - 1 and len(chosen) > 1:
                    n0, n1, nc, nf = infer(model, name, first_gray, grays[1], args.device)
                    negative_file = args.output_dir / f"negative_{name}_matches.npz"
                    np.savez_compressed(negative_file, points0=n0, points1=n1, confidence=nc)
                    negative_controls.append(
                        {
                            "model": name,
                            "rgb_sequence": chosen[0],
                            "ir_sequence": sequence,
                            "matches": len(n0),
                            "flags": nf,
                            "header_screen": header_diagnostic(
                                n0, n1, first_gray.shape[0], grays[1].shape[0]
                            ),
                            "matches_file": negative_file.name,
                            "matches_sha256": file_sha256(negative_file),
                        }
                    )
    project = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        project / "src/aero_ir/registration/detector_free.py",
        project / "scripts/audit_antiuav300_dense_registration.py",
        project / "src/aero_ir/data/antiuav.py",
    ]
    report = {
        "experiment": "pretrained_detector_free_train_screen_v1",
        "device": args.device,
        "elapsed_seconds": time.monotonic() - start,
        "fit_split": "none",
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "vendor_commit": commit,
        "weights_sha256": hashes,
        "torch_version": torch.__version__,
        "split_manifest_sha256": file_sha256(args.root / "label_new/train.json"),
        "inputs": provenance,
        "rows": rows,
        "known_translation_controls": controls,
        "unpaired_sequence_negative_controls": negative_controls,
        "summary": summary(rows),
        "protocol": {
            "long_side": args.long_side,
            "resize": "aspect preserved, floor to multiple of 8",
            "selection": "evenly spaced sorted train sequences, midpoint usable frame",
            "xoftr_coarse_threshold": 0.3,
            "xoftr_fine_threshold": 0.1,
            "loftr_coarse_threshold": 0.2,
            "reciprocity_tolerance_px": 2,
            "homography_ransac_threshold_px": 3,
            "homography_check": "random half held out matches, not independent GT",
            "annotations_used_for_matching": False,
            "independent_pixel_gt_available": False,
            "forced_xoftr_fine_placeholder": "suppressed and counted",
            "header_screen": "post-hoc exclusion if either point is in top 20%; reticles remain",
        },
        "sources": {str(p.relative_to(project)): file_sha256(p) for p in source_paths},
        "generator_training_eligible": "hold",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
