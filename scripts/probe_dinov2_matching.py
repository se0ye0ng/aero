#!/usr/bin/env python3
"""Frozen whole-image DINOv2 patch correspondence diagnostic; never pixel GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from aero_ir.registration.detector_free import inside_box
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_detector_free_screen import hull_fraction
from scripts.probe_detector_free_matching import read_pair, visual_review

VENDOR_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHT_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"
BASELINE_SHA256 = "1267cfe38eb0676b779704275f3e3939debb664b74bc7fe14598e7e6e292a6d8"
HEADERS = {"visible": 72, "infrared": 104}
PATCH = 14


def prepare(image, modality, condition):
    if modality not in HEADERS or condition not in ("original", "header_crop"):
        raise ValueError("unknown modality or input condition")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected RGB image")
    top = HEADERS[modality] if condition == "header_crop" else 0
    cropped = image[top:]
    h, w = cropped.shape[:2]
    nh, nw = h // PATCH * PATCH, w // PATCH * PATCH
    if min(nh, nw) < PATCH:
        raise ValueError("image smaller than a patch")
    if modality == "infrared":
        cropped = np.repeat(cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return resized, {
        "crop_top": top,
        "crop_shape": [h, w],
        "input_shape": [nh, nw],
        "scale_xy_to_original": [w / nw, h / nh],
        "patch_grid": [nh // PATCH, nw // PATCH],
    }


def patch_centres(shape):
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h // PATCH), np.arange(w // PATCH), indexing="ij")
    return np.stack(((xx + 0.5) * PATCH - 0.5, (yy + 0.5) * PATCH - 0.5), axis=-1).reshape(-1, 2)


def restore(points, metadata):
    return (
        (points + 0.5) * np.array(metadata["scale_xy_to_original"])
        - 0.5
        + np.array([0, metadata["crop_top"]])
    )


def extract(model, image):
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 255
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    features = model.forward_features((tensor - mean) / std)["x_norm_patchtokens"][0]
    if features.shape[0] != (image.shape[0] // PATCH) * (image.shape[1] // PATCH):
        raise ValueError("unexpected patch-token count")
    return features


def match_features(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("incompatible feature shapes")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite descriptors")
    if len(a) == 0 or len(b) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    similarities = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
    targets = similarities.argmax(1)
    sources = similarities.argmax(0)
    ids = torch.arange(len(a))
    keep = sources[targets] == ids
    # Zero descriptors are uninformative, even if arbitrary argmax ties agree.
    keep &= (a.norm(dim=1) > 1e-12) & (b[targets].norm(dim=1) > 1e-12)
    i, j = ids[keep], targets[keep]
    return i.numpy(), j.numpy(), similarities[i, j].numpy()


def metrics(p0, p1, centres0, centres1, boxes):
    a, b = inside_box(p0, boxes[0]), inside_box(p1, boxes[1])
    both = a & b
    counts = [
        int(inside_box(p, box).sum()) for p, box in zip((centres0, centres1), boxes, strict=True)
    ]
    return {
        "matches": len(p0),
        "source_box_matches": int(a.sum()),
        "target_box_matches": int(b.sum()),
        "both_box_matches": int(both.sum()),
        "target_patch_centres_in_box": counts,
        "rgb_hull_fraction": hull_fraction(p0[both], boxes[0]),
        "ir_hull_fraction": hull_fraction(p1[both], boxes[1]),
        "low_target_resolution": min(counts) < 4,
        "mutuality_is_construction_not_independent_replay": True,
    }


def synthetic_control(model, prepared, features):
    h, w = prepared.shape[:2]
    shift = np.array([28.0, -28.0])
    warped = cv2.warpAffine(prepared, np.array([[1.0, 0.0, 28.0], [0.0, 1.0, -28.0]]), (w, h))
    second = extract(model, warped)
    i, j, _ = match_features(features, second)
    centres = patch_centres((h, w))
    p, q = centres[i], centres[j]
    valid = ((p >= 42) & (p < np.array([w, h]) - 42)).all(1)
    valid &= ((q >= 42) & (q < np.array([w, h]) - 42)).all(1)
    errors = np.linalg.norm(q[valid] - p[valid] - shift, axis=1)
    return {
        "known_shift_model_pixels": shift.tolist(),
        "matches": len(i),
        "interior_matches": len(errors),
        "median_error_model_px": float(np.median(errors)) if len(errors) else None,
        "fractions_within_model_px": {
            str(t): float((errors <= t).mean()) if len(errors) else None for t in (3, 7, 14)
        },
    }


def load_model(external):
    vendor, weight = external / "dinov2", external / "dinov2_vits14_reg4_pretrain.pth"
    commit = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(vendor), "status", "--porcelain"], text=True
    ).strip()
    if commit != VENDOR_COMMIT or dirty or file_sha256(weight) != WEIGHT_SHA256:
        raise ValueError("DINOv2 vendor/weight provenance mismatch")
    sys.path.insert(0, str(vendor.resolve()))
    from dinov2.hub.backbones import dinov2_vits14_reg

    model = dinov2_vits14_reg(pretrained=False)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    return model.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300")
    )
    parser.add_argument("--external-root", type=Path, default=Path("experiments/external"))
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("experiments/detector_free_train16_01/report.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    if file_sha256(args.baseline_report) != BASELINE_SHA256:
        raise ValueError("baseline report hash differs from pinned comparison")
    baseline = json.loads(args.baseline_report.read_text())
    if baseline["evaluated_split"] != "train" or baseline["validation_or_test_access"] != "none":
        raise ValueError("baseline must be train-only")
    if len(baseline["inputs"]) != 16:
        raise ValueError("expected pinned 16-sequence baseline")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    model = load_model(args.external_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start, rows, controls, inputs, firsts, lasts = time.monotonic(), [], [], [], {}, {}
    with torch.inference_mode():
        for index, expected in enumerate(baseline["inputs"]):
            images, boxes, provenance = read_pair(args.root, expected["sequence_id"], 640)
            if provenance != expected:
                raise ValueError("decoded input provenance differs from pinned baseline")
            inputs.append(provenance)
            for condition in ("original", "header_crop"):
                prepared, metadata = zip(
                    *(
                        prepare(image, modality, condition)
                        for image, modality in zip(images, ("visible", "infrared"), strict=True)
                    ),
                    strict=True,
                )
                features = [extract(model, image) for image in prepared]
                centres = [
                    restore(patch_centres(image.shape[:2]), meta)
                    for image, meta in zip(prepared, metadata, strict=True)
                ]
                i, j, confidence = match_features(*features)
                p0, p1 = centres[0][i], centres[1][j]
                name = f"{index:03d}_{condition}_matches.npz"
                np.savez_compressed(
                    args.output_dir / name,
                    points0=p0,
                    points1=p1,
                    confidence=confidence,
                    patch_indices0=i,
                    patch_indices1=j,
                    patch_centres0=centres[0],
                    patch_centres1=centres[1],
                )
                rows.append(
                    {
                        "sequence_id": expected["sequence_id"],
                        "condition": condition,
                        "metadata": metadata,
                        "boxes_evaluation_only": [box.tolist() for box in boxes],
                        "prepared_sha256": [
                            hashlib.sha256(img.tobytes()).hexdigest() for img in prepared
                        ],
                        "matches_file": name,
                        "matches_sha256": file_sha256(args.output_dir / name),
                        "metrics": metrics(p0, p1, *centres, boxes),
                    }
                )
                visual_review(
                    args.output_dir / f"{index:03d}_{condition}_review.png",
                    images,
                    boxes,
                    p0,
                    p1,
                    confidence,
                    np.ones(len(p0), bool),
                    f"DINOv2 {condition}: patch MNN by construction, NOT pixel accuracy",
                )
                if index in (0, len(baseline["inputs"]) - 1):
                    controls.append(
                        {
                            "sequence_id": expected["sequence_id"],
                            "condition": condition,
                            **synthetic_control(model, prepared[0], features[0]),
                        }
                    )
                if index == 0:
                    firsts[condition] = (features[0], centres[0], boxes[0])
                lasts[condition] = (features[1], centres[1], boxes[1])
            print(f"{index + 1}/16 pairs", flush=True)
        negatives = {}
        for condition in firsts:
            a, ac, ab = firsts[condition]
            b, bc, bb = lasts[condition]
            i, j, confidence = match_features(a, b)
            name = f"negative_{condition}.npz"
            np.savez_compressed(
                args.output_dir / name, points0=ac[i], points1=bc[j], confidence=confidence
            )
            negatives[condition] = {
                "matches_file": name,
                "matches_sha256": file_sha256(args.output_dir / name),
                "metrics": metrics(ac[i], bc[j], ac, bc, [ab, bb]),
            }
    summaries = {}
    for condition in ("original", "header_crop"):
        ms = [row["metrics"] for row in rows if row["condition"] == condition]
        summaries[condition] = {
            "pairs": len(ms),
            "median_matches": float(np.median([m["matches"] for m in ms])),
            "pairs_any_both_box_match": sum(m["both_box_matches"] > 0 for m in ms),
            "pairs_four_both_box_matches": sum(m["both_box_matches"] >= 4 for m in ms),
            "pairs_four_and_ten_percent_hull": sum(
                m["both_box_matches"] >= 4
                and min(m["rgb_hull_fraction"], m["ir_hull_fraction"]) >= 0.1
                for m in ms
            ),
            "pairs_low_target_resolution": sum(m["low_target_resolution"] for m in ms),
            "total_both_box_matches": sum(m["both_box_matches"] for m in ms),
        }
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "scripts/probe_detector_free_matching.py",
        root / "scripts/analyze_detector_free_screen.py",
        root / "scripts/audit_antiuav300_dense_registration.py",
        root / "src/aero_ir/registration/detector_free.py",
        root / "src/aero_ir/utils/manifest.py",
    ]
    report = {
        "experiment": "dinov2_vits14_registers_whole_image_train16",
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "device": "cpu",
        "elapsed_seconds": time.monotonic() - start,
        "torch_version": torch.__version__,
        "vendor_commit": VENDOR_COMMIT,
        "weight_sha256": WEIGHT_SHA256,
        "baseline_report_sha256": file_sha256(args.baseline_report),
        "inputs": inputs,
        "sources": {str(path.relative_to(root)): file_sha256(path) for path in sources},
        "rows": rows,
        "synthetic": controls,
        "negative_controls": negatives,
        "summary": summaries,
        "protocol": {
            "normalization": "ImageNet mean/std",
            "rgb_input": "color",
            "ir_input": "grayscale repeated3",
            "feature": "last-layer x_norm_patchtokens, patch14, 4registers excluded",
            "matching": "cosine mutual nearest, no tuned threshold",
            "coordinates": (
                "half-pixel resize/crop restored to 640-derived coords; NOT native sensor pixels"
            ),
            "annotation_use": "evaluation only, never input cropping or match selection",
            "independent_pixel_gt": False,
        },
        "generator_training_eligible": "hold",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"summary": summaries, "synthetic": controls, "negative_controls": negatives}, indent=2
        )
    )


if __name__ == "__main__":
    main()
