#!/usr/bin/env python3
"""Train-only MIND loss-landscape diagnostic, not registration qualification.

No optimization, checkpoint mutation, validation/test access, or generator release.
The real-pair box proxy is NOT independent pixel correspondence ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import centre_grid, from_superfusion, map_boxes, sampling_map
from aero_ir.registration.mind import mind_descriptor
from aero_ir.registration.protocol_v2 import _edge_map
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.compare_antiuav300_registration_pilots import checkpoint_info

OFFSETS = (-8, -4, 0, 4, 8)
SCALES = (0.9, 1.0, 1.1)
MIN_PIXELS = 32
TEXTURE_SSD = 1e-6  # Fixed diagnostic exclusion, not calibrated confidence.


def candidates(field, pivot, scales=SCALES, offsets=OFFSETS):
    """Post-compose map with source-coordinate scale/translation; no source GT."""
    h, w = field.shape[-2:]
    base = sampling_map(field)
    maps, specs = [], []
    for scale in scales:
        for dy in offsets:
            for dx in offsets:
                maps.append(
                    pivot + scale * (base - pivot) + base.new_tensor([2 * dx / w, 2 * dy / h])
                )
                specs.append({"scale": scale, "dx_px": dx, "dy_px": dy})
    return torch.cat(maps), specs


def common_support(maps):
    """Identical target pixels for ALL candidates; no overlap-shrinking shortcut."""
    _, h, w, _ = maps.shape
    limit = maps.new_tensor([1 - 7 / w, 1 - 7 / h])  # 3px descriptor footprint.
    mask = (torch.isfinite(maps).all(-1) & (maps.abs() <= limit).all(-1)).all(0)
    mask[:3] = False
    mask[-3:] = False
    mask[:, :3] = False
    mask[:, -3:] = False
    return mask


def roi_mask(box, h, w, *, mode="expanded"):
    """Fixed target region: expanded context or exact annotation box."""
    yy, xx = torch.meshgrid(
        torch.arange(h, device=box.device) + 0.5,
        torch.arange(w, device=box.device) + 0.5,
        indexing="ij",
    )
    cx, cy, bw, bh = box.flatten() * box.new_tensor([w, h, w, h])
    if mode == "box":
        return ((xx - cx).abs() <= bw / 2) & ((yy - cy).abs() <= bh / 2)
    if mode != "expanded":
        raise ValueError("unknown ROI mode")
    return ((xx - cx).abs() <= torch.maximum(bw, bw.new_tensor(12))) & (
        (yy - cy).abs() <= torch.maximum(bh, bh.new_tensor(12))
    )


def score_candidates(source, target, maps, regions):
    """Compare MIND SSD and edge NCC under exactly the same fixed support."""
    sd, sv = mind_descriptor(source)
    td, tv = mind_descriptor(target)
    source_features = torch.cat((sd, _edge_map(source), sv), dim=1)
    target_edges = _edge_map(target)[0, 0]
    mask = common_support(maps)
    # Flat patches do not supply evidence. Exclude if ANY candidate lands on
    # a flat source patch, so every candidate is scored on the same pixels.
    source_texture = F.grid_sample(sv.expand(len(maps), -1, -1, -1), maps, align_corners=False)[
        :, 0
    ]
    mask &= (tv[0, 0] > TEXTURE_SSD) & (source_texture > TEXTURE_SSD).all(0)
    scores = {name: {"mind": [], "edge_ncc": []} for name in regions}
    supports = {name: mask & region for name, region in regions.items()}
    counts = {name: int(region.sum()) for name, region in supports.items()}
    for batch in maps.split(8):
        warped = F.grid_sample(
            source_features.expand(len(batch), -1, -1, -1), batch, align_corners=False
        )
        errors = (warped[:, :8] - td).square().mean(1)
        for name, support in supports.items():
            if counts[name] < MIN_PIXELS:
                continue
            scores[name]["mind"].extend(errors[:, support].mean(1).tolist())
            a, b = warped[:, 8, support], target_edges[support]
            a, b = a - a.mean(1, keepdim=True), b - b.mean()
            denom = a.square().sum(1).sqrt() * b.square().sum().sqrt()
            losses = 1 - (a * b).sum(1) / denom.clamp_min(1e-12)
            # Constant edge maps are not valid NCC observations.
            scores[name]["edge_ncc"].extend(
                float(value) if float(d) > 1e-10 else None
                for value, d in zip(losses, denom, strict=True)
            )
    return scores, counts, supports


def box_iou(predicted, truth):
    low, high = predicted[:, :2] - predicted[:, 2:] / 2, predicted[:, :2] + predicted[:, 2:] / 2
    t_low, t_high = truth[:, :2] - truth[:, 2:] / 2, truth[:, :2] + truth[:, 2:] / 2
    intersection = (torch.minimum(high, t_high) - torch.maximum(low, t_low)).clamp_min(0).prod(1)
    return intersection / (
        predicted[:, 2:].prod(1) + truth[:, 2:].prod(1) - intersection
    ).clamp_min(1e-8)


def best_index(values, baseline):
    """Do not turn ties/constant descriptors into claimed displacement evidence."""
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return None
    if max(values) - min(values) <= 1e-8:
        return None
    minimum = min(values)
    if values[baseline] <= minimum + 1e-8:
        return baseline
    return int(np.argmin(values))


def synthetic_check(image):
    """Known translation + contrast inversion; verifies implementation, not RGB/IR."""
    h, w = image.shape[-2:]
    zero = image.new_zeros((1, 2, h, w))
    maps, specs = candidates(zero, image.new_zeros((1, 1, 1, 2)), scales=(1.0,))
    truth = next(i for i, s in enumerate(specs) if s["dx_px"] == 4 and s["dy_px"] == -4)
    target = 1 - F.grid_sample(image, maps[truth : truth + 1], align_corners=False)
    # Exclude pixels whose target descriptor footprint touches generated padding.
    region = torch.zeros((h, w), dtype=torch.bool, device=image.device)
    region[16:-16, 16:-16] = True
    scores, counts, _ = score_candidates(image, target, maps, {"interior": region})
    result = {"truth": specs[truth], "supported_pixels": counts["interior"]}
    for method, values in scores["interior"].items():
        selected = best_index(values, truth)
        result[method] = {
            "selected": specs[selected] if selected is not None else None,
            "exact_recovery": selected == truth,
            "loss_at_truth": values[truth] if values else None,
        }
    return result


def real_check(
    source,
    target,
    source_box,
    target_box,
    field,
    *,
    scales=SCALES,
    offsets=OFFSETS,
    roi_mode="expanded",
):
    _, _, h, w = field.shape
    # Use predicted enclosure center, NOT source annotation, for the search pivot.
    pivot = (2 * map_boxes(target_box, field)[:, :2] - 1).reshape(1, 1, 1, 2)
    maps, specs = candidates(field, pivot, scales=scales, offsets=offsets)
    baseline = next(i for i, s in enumerate(specs) if s == {"scale": 1.0, "dx_px": 0, "dy_px": 0})
    roi = roi_mask(target_box, h, w, mode=roi_mode)
    scores, counts, _ = score_candidates(source, target, maps, {"roi": roi, "background": ~roi})
    fields = (maps - centre_grid(field)).permute(0, 3, 1, 2)
    ious = box_iou(map_boxes(target_box.expand(len(maps), -1), fields), source_box).tolist()
    summaries = {}
    for region, methods in scores.items():
        summaries[region] = {}
        for method, values in methods.items():
            selected = best_index(values, baseline)
            summaries[region][method] = {
                "selected": specs[selected] if selected is not None else None,
                "selected_iou": ious[selected] if selected is not None else None,
                "delta_iou": ious[selected] - ious[baseline] if selected is not None else None,
                "loss_at_baseline": values[baseline] if values else None,
                "minimum_loss": values[selected] if selected is not None else None,
            }
    return {
        "baseline_iou": ious[baseline],
        "best_candidate_box_iou_oracle": max(ious),
        "counts": counts,
        "region_pixels": {"roi": int(roi.sum()), "background": int((~roi).sum())},
        "selection": summaries,
        "candidates": specs,
        "candidate_box_ious": ious,
        "losses": scores,
    }


def summarize(rows):
    results = {}
    for direction in ("ir_to_rgb_points", "rgb_to_ir_points"):
        subset = [r for r in rows if r["direction"] == direction]
        results[direction] = {}
        for region in ("roi", "background"):
            results[direction][region] = {}
            for method in ("mind", "edge_ncc"):
                deltas = [r["real"]["selection"][region][method]["delta_iou"] for r in subset]
                valid = [d for d in deltas if d is not None]
                results[direction][region][method] = {
                    "n_total": len(subset),
                    "n_valid": len(valid),
                    "mean_delta_box_iou": float(np.mean(valid)) if valid else None,
                    "median_delta_box_iou": float(np.median(valid)) if valid else None,
                    "improved_gt_0_01": sum(d > 0.01 for d in valid),
                    "worsened_gt_0_01": sum(d < -0.01 for d in valid),
                    "unchanged_within_0_01": sum(abs(d) <= 0.01 for d in valid),
                }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--search", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--roi-mode", choices=("expanded", "box"), default="expanded")
    args = parser.parse_args()
    offsets = (-2, -1, 0, 1, 2) if args.search == "fine" else OFFSETS
    scales = (0.98, 1.0, 1.02) if args.search == "fine" else SCALES
    if args.output_dir.exists():
        parser.error("choose a fresh output directory; previous results are immutable")
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
    chosen = [shards[i] for i in np.linspace(0, len(shards) - 1, args.sequences, dtype=int)]
    model = load_superfusion_matcher(args.checkpoint, torch.device("cpu")).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, rows, synthetic, inputs = time.monotonic(), [], [], []
    with torch.inference_mode():
        for index, shard in enumerate(chosen):
            folder = args.cache_root / "shards" / shard["sequence_id"]
            position = shard["pairs"] // 2
            arrays = {
                k: np.array(np.load(folder / f"{k}.npy", mmap_mode="r")[position])
                for k in ("visible", "infrared", "source_boxes", "target_boxes")
            }
            inputs.append(
                {
                    "sequence_id": shard["sequence_id"],
                    "cache_position": position,
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
                synthetic.append(
                    {
                        "sequence_id": shard["sequence_id"],
                        "modality": name,
                        **synthetic_check(image),
                    }
                )
            for direction, model_direction, source, target, sb, tb in (
                ("ir_to_rgb_points", "visible_to_infrared", vis, ir, vb, ib),
                ("rgb_to_ir_points", "infrared_to_visible", ir, vis, ib, vb),
            ):
                field = from_superfusion(model(ir, vis, direction=model_direction))
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
                            scales=scales,
                            offsets=offsets,
                            roi_mode=args.roi_mode,
                        ),
                    }
                )
            print(f"{index + 1}/{len(chosen)} train sequences evaluated", flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "src/aero_ir/registration/mind.py",
        root / "src/aero_ir/registration/geometry.py",
        root / "src/aero_ir/registration/protocol_v2.py",
        root / "src/aero_ir/registration/superfusion.py",
        root / "scripts/compare_antiuav300_registration_pilots.py",
        root / "src/aero_ir/utils/manifest.py",
    ]
    report = {
        "experiment": "mind_2d_train_only_loss_landscape_v1",
        "elapsed_seconds": time.monotonic() - started,
        "device": "cpu",
        "torch_version": torch.__version__,
        "checkpoint": checkpoint,
        "cache_manifest_file_sha256": file_sha256(manifest_path),
        "inputs": inputs,
        "sources": {str(p.relative_to(root)): file_sha256(p) for p in sources},
        "protocol": {
            "offsets_px": offsets,
            "scales": scales,
            "search": args.search,
            "roi_mode": args.roi_mode,
            "min_pixels": MIN_PIXELS,
            "texture_ssd_threshold": TEXTURE_SSD,
            "fixed_common_support_across_candidates": True,
            "selection": "equally spaced sorted train sequences, midpoint usable cache pair",
            "independent_pixel_correspondence_gt": False,
            "real_metric": "box IoU proxy, not pixel accuracy",
            "descriptor": "2D MIND-style axial r1/r2, uniform 3x3 SSD, grayscale",
        },
        "synthetic": synthetic,
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
