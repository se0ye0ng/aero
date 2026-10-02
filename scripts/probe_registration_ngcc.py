"""Frozen-v6 train-only gradient similarity landscape, without training."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import centre_grid, from_superfusion, map_boxes
from aero_ir.registration.ngcc import METHODS, scores
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


def fixed_support(maps, source_header=0.0, target_header=0.0):
    h, w = maps.shape[1:3]
    support = common_support(maps)
    # Sobel uses adjacent target pixels, so erode validity of sampled source.
    # Sample no removed source pixel, including bilinear interpolation footprint.
    first_source_center = (np.ceil(source_header * h) + 0.5) / h
    valid = (maps[..., 1] >= 2 * first_source_center - 1).all(0)
    support &= valid
    # A scored Sobel stencil requires every neighbor's mapped sample to be valid,
    # not just its center (important near discontinuities and fold boundaries).
    support = F.avg_pool2d(support.float()[None, None], 3, 1, 1)[0, 0] >= 1
    support[: int(np.ceil(target_header * h)) + 1] = False
    return support


def evaluate(source, target, maps, regions, source_header=0.0, target_header=0.0):
    support = fixed_support(maps, source_header, target_header)
    masks = {key: support & value for key, value in regions.items()}
    values = {key: {method: [] for method in METHODS} for key in regions}
    for batch in maps.split(8):
        warped = F.grid_sample(source.expand(len(batch), -1, -1, -1), batch, align_corners=False)
        for key, mask in masks.items():
            if mask.sum() < 32:
                continue
            losses, valid = scores(warped, target, mask)
            for method in METHODS:
                values[key][method].extend(
                    float(x) if bool(ok) and torch.isfinite(x) else None
                    for x, ok in zip(losses[method], valid, strict=True)
                )
    return values, {key: int(value.sum()) for key, value in masks.items()}


def synthetic(image):
    h, w = image.shape[-2:]
    maps, specs = candidates(
        image.new_zeros(1, 2, h, w), image.new_zeros(1, 1, 1, 2), scales=(1.0,), offsets=(-4, 0, 4)
    )
    truth = specs.index({"scale": 1.0, "dx_px": 4, "dy_px": -4})
    baseline = specs.index({"scale": 1.0, "dx_px": 0, "dy_px": 0})
    base = F.grid_sample(image, maps[truth : truth + 1], align_corners=False)
    roi = torch.zeros(h, w, dtype=torch.bool)
    roi[16:-16, 16:-16] = True
    result = {}
    for name, target in (("normal", base), ("inverted", 1 - base)):
        values, _ = evaluate(image, target, maps, {"interior": roi})
        result[name] = {m: best_index(v, baseline) == truth for m, v in values["interior"].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output exists; choose an immutable fresh run directory")
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
        raise ValueError("invalid train-only cache provenance")
    shards = sorted(manifest["shards"], key=lambda s: s["sequence_id"])
    if not 1 <= args.sequences <= len(shards):
        parser.error("invalid sequence count")
    chosen = [shards[i] for i in np.linspace(0, len(shards) - 1, args.sequences, dtype=int)]
    model = load_superfusion_matcher(args.checkpoint, torch.device("cpu")).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, rows, controls, inputs = time.monotonic(), [], [], []
    with torch.inference_mode():
        for index, shard in enumerate(chosen):
            position = shard["pairs"] // 2
            folder = args.cache_root / "shards" / shard["sequence_id"]
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
            controls.extend(synthetic(image) for image in (vis, ir))
            for direction, md, source, target, sb, tb, sh, th in (
                ("ir_to_rgb_points", "visible_to_infrared", vis, ir, vb, ib, 0.2, 104 / 512),
                ("rgb_to_ir_points", "infrared_to_visible", ir, vis, ib, vb, 104 / 512, 0.2),
            ):
                field = from_superfusion(model(ir, vis, direction=md))
                h, w = field.shape[-2:]
                pivot = (2 * map_boxes(tb, field)[:, :2] - 1).reshape(1, 1, 1, 2)
                maps, specs = candidates(
                    field, pivot, scales=(0.98, 1.0, 1.02), offsets=(-2, -1, 0, 1, 2)
                )
                baseline = specs.index({"scale": 1.0, "dx_px": 0, "dy_px": 0})
                fields = (maps - centre_grid(field)).permute(0, 3, 1, 2)
                ious = box_iou(map_boxes(tb.expand(len(maps), -1), fields), sb).tolist()
                expanded, box = roi_mask(tb, h, w), roi_mask(tb, h, w, mode="box")
                for variant, hs, ht in (("original", 0.0, 0.0), ("header_excluded", sh, th)):
                    values, counts = evaluate(
                        source,
                        target,
                        maps,
                        {"expanded": expanded, "box": box, "background": ~expanded},
                        hs,
                        ht,
                    )
                    selection = {}
                    for region, methods in values.items():
                        selection[region] = {}
                        for method, vals in methods.items():
                            selected = best_index(vals, baseline)
                            selection[region][method] = {
                                "candidate": selected,
                                "delta_iou": None
                                if selected is None
                                else ious[selected] - ious[baseline],
                            }
                    rows.append(
                        {
                            "sequence_id": shard["sequence_id"],
                            "direction": direction,
                            "variant": variant,
                            "baseline_iou": ious[baseline],
                            "ious": ious,
                            "specs": specs,
                            "scores": values,
                            "counts": counts,
                            "selection": selection,
                        }
                    )
            print(f"{index + 1}/{len(chosen)}", flush=True)
    summary = {}
    for row in rows:
        for region, methods in row["selection"].items():
            for method, selection in methods.items():
                key = "/".join((row["variant"], row["direction"], region, method))
                summary.setdefault(key, []).append(selection["delta_iou"])
    summary = {
        k: {
            "n_total": len(v),
            "n_valid": len(a := [x for x in v if x is not None]),
            "mean_delta_iou": float(np.mean(a)) if a else None,
            "median_delta_iou": float(np.median(a)) if a else None,
            "improved_gt_0_01": sum(x > 0.01 for x in a),
            "worsened_gt_0_01": sum(x < -0.01 for x in a),
        }
        for k, v in summary.items()
    }
    root = Path(__file__).resolve().parents[1]
    sources = (
        "scripts/probe_registration_ngcc.py",
        "src/aero_ir/registration/ngcc.py",
        "scripts/probe_registration_mind.py",
        "src/aero_ir/registration/geometry.py",
    )
    report = {
        "experiment": "ngcc_train_only_landscape_v1",
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": checkpoint,
        "cache_manifest_file_sha256": file_sha256(manifest_path),
        "sources": {p: file_sha256(root / p) for p in sources},
        "inputs": inputs,
        "protocol": {
            "fit_split": "train",
            "validation_or_test_access": "none",
            "independent_pixel_gt": False,
            "header_rgb": 0.2,
            "header_ir": 104 / 512,
            "gradient": "Sobel/8 after image warp",
            "epsilon": 0.001,
            "energy_threshold": 1e-8,
            "min_pixels": 32,
        },
        "synthetic": controls,
        "rows": rows,
        "summary": summary,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
