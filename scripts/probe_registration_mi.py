#!/usr/bin/env python3
"""Train-only fixed-support MI candidate screen; no training or qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import centre_grid, from_superfusion, map_boxes
from aero_ir.registration.mutual_information import mutual_information
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.compare_antiuav300_registration_pilots import checkpoint_info
from scripts.probe_registration_mind import (
    best_index,
    box_iou,
    candidates,
    common_support,
    roi_mask,
)

MIN_PIXELS = 128
MAX_SAMPLES = 4096
MIN_STD = 1e-3
HEADER_FRACTIONS = {"visible": 72 / 360, "infrared": 104 / 512}
METHODS = ("gaussian_mi", "histogram_mi")


def grayscale(image):
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError("expected one or three channels")
    return (image * image.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(
        1, keepdim=True
    )


def masks_for_candidates(maps, source_header, target_header):
    """Common support; excluded inputs are never histogram observations."""
    mask = common_support(maps)
    _, h, _, _ = maps.shape
    yy = (torch.arange(h, device=maps.device) + 0.5) / h
    without_header = mask & (yy[:, None] >= target_header)
    # Sampling at the continuous crop boundary can still interpolate a rejected
    # row. Require the centre of the first wholly accepted source pixel.
    first_valid_source_centre = (math.ceil(h * source_header - 0.5) + 0.5) / h
    without_header &= (((maps[..., 1] + 1) / 2) >= first_valid_source_centre).all(0)
    return {"original": mask, "header_excluded": without_header}


def score_maps(source, target, maps, regions, *, source_header=0.0, target_header=0.0):
    source, target = grayscale(source), grayscale(target)
    warped = F.grid_sample(source.expand(len(maps), -1, -1, -1), maps, align_corners=False)[:, 0]
    result = {}
    masks = masks_for_candidates(maps, source_header, target_header)
    for condition, common in masks.items():
        result[condition] = {}
        for name, region in regions.items():
            indices = (common & region).flatten().nonzero().flatten()
            supported = len(indices)
            # Deterministic evenly spaced spatial subset, identical for every candidate.
            if supported > MAX_SAMPLES:
                indices = indices[torch.linspace(0, supported - 1, MAX_SAMPLES).long()]
            row = {"supported_pixels": supported, "sampled_pixels": len(indices), "losses": {}}
            result[condition][name] = row
            if len(indices) < MIN_PIXELS:
                row["abstention"] = "insufficient_support"
                continue
            a = warped.flatten(1)[:, indices]
            b = target[0, 0].flatten()[indices].expand_as(a)
            if b[0].std(unbiased=False) <= MIN_STD or (a.std(1, unbiased=False) <= MIN_STD).any():
                row["abstention"] = "flat_input_in_at_least_one_candidate"
                continue
            row["abstention"] = None
            for method in METHODS:
                values = []
                for aa, bb in zip(a.split(8), b.split(8), strict=True):
                    values.extend(
                        (-mutual_information(aa, bb, hard=method == "histogram_mi")).tolist()
                    )
                row["losses"][method] = values
    return result


def real_check(source, target, source_box, target_box, field, *, source_header=0, target_header=0):
    _, _, h, w = field.shape
    pivot = (2 * map_boxes(target_box, field)[:, :2] - 1).reshape(1, 1, 1, 2)
    maps, specs = candidates(field, pivot, scales=(0.98, 1.0, 1.02), offsets=(-2, -1, 0, 1, 2))
    baseline = specs.index({"scale": 1.0, "dx_px": 0, "dy_px": 0})
    expanded = roi_mask(target_box, h, w)
    regions = {
        "expanded": expanded,
        "box": roi_mask(target_box, h, w, mode="box"),
        "background": ~expanded,
    }
    scores = score_maps(
        source, target, maps, regions, source_header=source_header, target_header=target_header
    )
    fields = (maps - centre_grid(field)).permute(0, 3, 1, 2)
    ious = box_iou(map_boxes(target_box.expand(len(maps), -1), fields), source_box).tolist()
    for condition in scores.values():
        for row in condition.values():
            row["selection"] = {}
            for method in METHODS:
                values = row["losses"].get(method, [])
                chosen = best_index(values, baseline)
                row["selection"][method] = {
                    "candidate": specs[chosen] if chosen is not None else None,
                    "delta_box_iou": ious[chosen] - ious[baseline] if chosen is not None else None,
                    "loss_at_baseline": values[baseline] if values else None,
                }
    return {
        "baseline_iou": ious[baseline],
        "oracle_iou": max(ious),
        "candidate_box_ious": ious,
        "candidates": specs,
        "regions": scores,
    }


def synthetic_check(image):
    """Known integer translation with three same-modality appearance transforms."""
    image = grayscale(image)
    h, w = image.shape[-2:]
    maps, specs = candidates(
        image.new_zeros(1, 2, h, w), image.new_zeros(1, 1, 1, 2), scales=(1.0,), offsets=(-4, 0, 4)
    )
    truth = specs.index({"scale": 1.0, "dx_px": 4, "dy_px": -4})
    identity = specs.index({"scale": 1.0, "dx_px": 0, "dy_px": 0})
    shifted = F.grid_sample(image, maps[truth : truth + 1], align_corners=False)
    region = torch.zeros((h, w), dtype=torch.bool)
    region[12:-12, 12:-12] = True
    controls = {}
    for name, target in (
        ("same", shifted),
        ("inverted", 1 - shifted),
        ("gamma2", shifted.square()),
    ):
        row = score_maps(image, target, maps, {"interior": region})["original"]["interior"]
        controls[name] = {"abstention": row["abstention"], "methods": {}}
        for method in METHODS:
            values = row["losses"].get(method, [])
            chosen = best_index(values, identity)
            controls[name]["methods"][method] = {
                "exact_recovery": chosen == truth,
                "selected": specs[chosen] if chosen is not None else None,
            }
    return controls


def summarize(rows):
    result = {}
    for direction in ("ir_to_rgb_points", "rgb_to_ir_points"):
        subset = [r for r in rows if r["direction"] == direction]
        result[direction] = {}
        for condition in ("original", "header_excluded"):
            result[direction][condition] = {}
            for region in ("expanded", "box", "background"):
                result[direction][condition][region] = {}
                for method in METHODS:
                    ds = [
                        r["real"]["regions"][condition][region]["selection"][method][
                            "delta_box_iou"
                        ]
                        for r in subset
                    ]
                    valid = [d for d in ds if d is not None]
                    result[direction][condition][region][method] = {
                        "n_total": len(ds),
                        "n_valid": len(valid),
                        "mean_delta_box_iou": float(np.mean(valid)) if valid else None,
                        "median_delta_box_iou": float(np.median(valid)) if valid else None,
                        "improved_gt_0_01": sum(d > 0.01 for d in valid),
                        "worsened_gt_0_01": sum(d < -0.01 for d in valid),
                        "unchanged_within_0_01": sum(abs(d) <= 0.01 for d in valid),
                    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output already exists; use a fresh directory")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    manifest_path = args.cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = checkpoint_info(args.checkpoint)
    if (
        manifest["fit_split"] != "train"
        or manifest["validation_or_test_access"] != "none"
        or canonical_hash({k: v for k, v in manifest.items() if k != "cache_manifest_sha256"})
        != manifest["cache_manifest_sha256"]
        or checkpoint["training_metadata"]["cache_manifest_sha256"]
        != manifest["cache_manifest_sha256"]
    ):
        raise ValueError("invalid or mismatching train-only cache")
    shards = sorted(manifest["shards"], key=lambda s: s["sequence_id"])
    if not 1 <= args.sequences <= len(shards):
        parser.error("invalid sequence count")
    selected = [shards[i] for i in np.linspace(0, len(shards) - 1, args.sequences, dtype=int)]
    model = load_superfusion_matcher(args.checkpoint, torch.device("cpu")).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start, rows, controls, inputs = time.monotonic(), [], [], []
    with torch.inference_mode():
        for index, shard in enumerate(selected):
            folder, pos = args.cache_root / "shards" / shard["sequence_id"], shard["pairs"] // 2
            arrays = {
                k: np.array(np.load(folder / f"{k}.npy", mmap_mode="r")[pos])
                for k in ("visible", "infrared", "source_boxes", "target_boxes")
            }
            inputs.append(
                {
                    "sequence_id": shard["sequence_id"],
                    "cache_position": pos,
                    "selected_array_sha256": {
                        k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()
                    },
                }
            )
            vis, ir = [
                torch.from_numpy(arrays[k]).permute(2, 0, 1)[None].float() / 255
                for k in ("visible", "infrared")
            ]
            vb, ib = [
                torch.from_numpy(arrays[k])[None].float() for k in ("source_boxes", "target_boxes")
            ]
            for name, image in (("visible", vis), ("infrared", ir)):
                controls.append(
                    {
                        "sequence_id": shard["sequence_id"],
                        "modality": name,
                        "controls": synthetic_check(image),
                    }
                )
            for direction, md, source, target, sb, tb, sm, tm in (
                ("ir_to_rgb_points", "visible_to_infrared", vis, ir, vb, ib, "visible", "infrared"),
                ("rgb_to_ir_points", "infrared_to_visible", ir, vis, ib, vb, "infrared", "visible"),
            ):
                field = from_superfusion(model(ir, vis, direction=md))
                rows.append(
                    {
                        "sequence_id": shard["sequence_id"],
                        "direction": direction,
                        "real": real_check(
                            source,
                            target,
                            sb,
                            tb,
                            field,
                            source_header=HEADER_FRACTIONS[sm],
                            target_header=HEADER_FRACTIONS[tm],
                        ),
                    }
                )
            print(f"{index + 1}/{len(selected)} sequences", flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "src/aero_ir/registration/mutual_information.py",
        root / "scripts/probe_registration_mind.py",
        root / "src/aero_ir/registration/geometry.py",
        root / "src/aero_ir/registration/superfusion.py",
        root / "scripts/compare_antiuav300_registration_pilots.py",
        root / "src/aero_ir/utils/manifest.py",
    ]
    report = {
        "experiment": "mi_train_only_loss_landscape_v1",
        "elapsed_seconds": time.monotonic() - start,
        "device": "cpu",
        "torch_version": torch.__version__,
        "checkpoint": checkpoint,
        "cache_manifest_file_sha256": file_sha256(manifest_path),
        "inputs": inputs,
        "sources": {str(p.relative_to(root)): file_sha256(p) for p in sources},
        "protocol": {
            "bins": 16,
            "bandwidth": 0.5 / 15,
            "min_pixels": MIN_PIXELS,
            "max_samples": MAX_SAMPLES,
            "min_std": MIN_STD,
            "header_fractions": HEADER_FRACTIONS,
            "fixed_common_support": True,
            "independent_pixel_correspondence_gt": False,
            "selection": "equally spaced sorted train sequences, midpoint usable pair",
            "estimator": "Gaussian-Parzen soft histogram, not Mattes MI; hard histogram reference",
            "overlay_policy": (
                "sample exclusion only; frozen v6 predictor still sees original inputs"
            ),
        },
        "synthetic": controls,
        "rows": rows,
        "summary": summarize(rows),
        "generator_training_eligible": "hold",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
