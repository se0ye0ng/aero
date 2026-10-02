#!/usr/bin/env python3
"""Bounded train-only diagnosis of independent reciprocal predictions.

No optimization or GT-derived transforms. Compare the original learned pair,
numerical inversion of either native field, and a label-free affine projection.
Small-cycle results from constructed inverses are algebraic, not accuracy evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from aero_ir.registration.geometry import (
    centre_grid,
    cycle_field,
    from_superfusion,
    map_points,
    pixel_norm,
    sampling_map,
    valid_support,
)
from aero_ir.registration.qualification_v4 import direction_statistics
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.compare_antiuav300_registration_pilots import (
    DIRECTIONS,
    checkpoint_info,
    source_hashes,
    summary_for,
)


def invert_field(field, seed, iterations=20):
    """Damped Newton inverse seeded by the learned reciprocal; no GT access.

    Border extension is a numerical definition only: cycle/support statistics
    still exclude out-of-domain points. Non-injective fields need not invert.
    """
    n, _, h, w = field.shape
    targets = centre_grid(field).expand(n, -1, -1, -1).flatten(1, 2)
    x = sampling_map(seed).flatten(1, 2).clone()
    ex, ey = x.new_tensor([2 / w, 0]), x.new_tensor([0, 2 / h])
    for _ in range(iterations):
        error = map_points(field, x) - targets
        dx = (map_points(field, x + ex) - map_points(field, x - ex)) * (w / 4)
        dy = (map_points(field, x + ey) - map_points(field, x - ey)) * (h / 4)
        a, b, c, d = dx[..., 0], dy[..., 0], dx[..., 1], dy[..., 1]
        det = a * d - b * c
        usable = torch.isfinite(det) & (det.abs() > 1e-4)
        safe = torch.where(usable, det, torch.ones_like(det))
        update = torch.stack(
            (
                (d * error[..., 0] - b * error[..., 1]) / safe,
                (a * error[..., 1] - c * error[..., 0]) / safe,
            ),
            -1,
        )
        update = torch.where(usable[..., None], update, torch.zeros_like(update))
        update = update.clamp(-0.05, 0.05)
        best_error = error.square().sum(-1)
        best = x
        for scale in (1.0, 0.5, 0.25):
            candidate = x - scale * update
            candidate_error = (map_points(field, candidate) - targets).square().sum(-1)
            improved = torch.isfinite(candidate_error) & (candidate_error < best_error)
            best = torch.where(improved[..., None], candidate, best)
            best_error = torch.where(improved, candidate_error, best_error)
        x = best
    inverse = (x - targets).reshape(n, h, w, 2).permute(0, 3, 1, 2)
    residual = pixel_norm(map_points(field, x) - targets, h, w)
    supported = valid_support(x, h, w)
    return inverse, {
        "iterations": iterations,
        "inverse_lattice_residual_p95_px": float(torch.quantile(residual, 0.95)),
        "supported_converged_fraction_0_1px": float(((residual <= 0.1) & supported).float().mean()),
    }


def affine_projection(field):
    """Fit a forward affine map to uniform native-field samples, not boxes."""
    n, _, h, w = field.shape
    grid = centre_grid(field).expand(n, -1, -1, -1)
    coords = grid[:, ::8, ::8].flatten(1, 2).double()
    mapped = sampling_map(field)[:, ::8, ::8].flatten(1, 2).double()
    basis = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    # Small full-rank normal equation; same implementation on CPU and CUDA.
    coeff = torch.linalg.solve(basis.transpose(1, 2) @ basis, basis.transpose(1, 2) @ mapped).to(
        field.dtype
    )
    matrix, offset = coeff[:, :2], coeff[:, 2:]
    if (torch.linalg.det(matrix) <= 1e-4).any():
        raise ValueError("affine projection is singular or orientation reversing")
    points = grid.flatten(1, 2)
    forward = points @ matrix + offset
    reverse = (points - offset) @ torch.linalg.inv(matrix)
    return (
        (forward - points).reshape(n, h, w, 2).permute(0, 3, 1, 2),
        (reverse - points).reshape(n, h, w, 2).permute(0, 3, 1, 2),
    )


def spatial_cycle(first, second):
    residual, valid = cycle_field(first, second)
    h, w = first.shape[2:]
    errors = pixel_norm(residual, h, w)
    interior = torch.zeros_like(valid)
    interior[:, 16:-16, 16:-16] = True
    result = {}
    for name, region in (("interior", interior), ("border16", ~interior)):
        values = errors[valid & region]
        result[name] = {
            "count": len(values),
            "p95_px": (float(torch.quantile(values, 0.95)) if len(values) else None),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    checkpoint = checkpoint_info(args.checkpoint)
    manifest_path = args.cache_root / "manifest.json"
    manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    unsigned = {k: v for k, v in manifest.items() if k != "cache_manifest_sha256"}
    if (
        canonical_hash(unsigned) != manifest["cache_manifest_sha256"]
        or checkpoint["training_metadata"]["cache_manifest_sha256"]
        != manifest["cache_manifest_sha256"]
        or manifest["fit_split"] != "train"
        or manifest["validation_or_test_access"] != "none"
    ):
        raise ValueError("invalid or mismatching train-only cache")
    shards = sorted(manifest["shards"], key=lambda s: s["sequence_id"])
    if not 1 <= args.sequences <= len(shards):
        parser.error("invalid sequence count")
    selected = [shards[i] for i in np.linspace(0, len(shards) - 1, args.sequences, dtype=int)]
    project = Path(__file__).resolve().parents[1]
    sources = {
        **source_hashes(project),
        str(Path(__file__).relative_to(project)): file_sha256(__file__),
    }
    model = load_superfusion_matcher(args.checkpoint, device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, diagnostics, inputs = {}, [], []
    started = time.monotonic()
    with torch.inference_mode():
        for index, shard in enumerate(selected):
            folder = args.cache_root / "shards" / shard["sequence_id"]
            position = shard["pairs"] // 2
            arrays = {
                key: np.array(np.load(folder / f"{key}.npy", mmap_mode="r")[position])
                for key in ("visible", "infrared", "source_boxes", "target_boxes")
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

            def image(key, arrays=arrays):
                return torch.from_numpy(arrays[key]).permute(2, 0, 1)[None].to(device).float() / 255

            vis, ir = image("visible"), image("infrared")
            vb = torch.from_numpy(arrays["source_boxes"])[None].to(device)
            ib = torch.from_numpy(arrays["target_boxes"])[None].to(device)
            f = from_superfusion(model(ir, vis, direction="visible_to_infrared"))
            r = from_superfusion(model(ir, vis, direction="infrared_to_visible"))
            inv_f, diag_f = invert_field(f, r)
            inv_r, diag_r = invert_field(r, f)
            variants = {
                "learned_pair": (f, r),
                "keep_ir_to_rgb_invert": (f, inv_f),
                "keep_rgb_to_ir_invert": (inv_r, r),
            }
            rejected = {}
            for label, field in (("affine_ir_to_rgb", f), ("affine_rgb_to_ir", r)):
                try:
                    a, b = affine_projection(field)
                    variants[label] = (a, b) if label == "affine_ir_to_rgb" else (b, a)
                except ValueError as exc:
                    rejected[label] = str(exc)
            for label, (first, second) in variants.items():
                a = direction_statistics(first, second, ib, vb)[0]
                b = direction_statistics(second, first, vb, ib)[0]
                for item in (a, b):
                    item["global_nonpositive_jacobian_fraction"] = (
                        1 - item["positive_jacobian_fraction"] if item["finite_field"] else None
                    )
                rows.setdefault(label, []).append(
                    {
                        "sequence_id": shard["sequence_id"],
                        "frame_index": position,
                        "index_semantics": "cache_position_not_video_frame",
                        DIRECTIONS[0]: a,
                        DIRECTIONS[1]: b,
                    }
                )
            diagnostics.append(
                {
                    "sequence_id": shard["sequence_id"],
                    "invert_ir_to_rgb": diag_f,
                    "invert_rgb_to_ir": diag_r,
                    "rejected": rejected,
                    "original_spatial_cycle": {
                        "ir_to_rgb": spatial_cycle(f, r),
                        "rgb_to_ir": spatial_cycle(r, f),
                    },
                }
            )
            print(
                f"{index + 1}/{len(selected)} sequences, elapsed {time.monotonic() - started:.1f}s",
                flush=True,
            )
    if file_sha256(args.checkpoint) != checkpoint["sha256"]:
        raise RuntimeError("checkpoint changed")
    if file_sha256(manifest_path) != manifest_sha:
        raise RuntimeError("cache manifest changed")
    if any(file_sha256(project / name) != digest for name, digest in sources.items()):
        raise RuntimeError("source changed")
    report = {
        "kind": "train_only_inverse_feasibility_probe",
        "checkpoint": checkpoint,
        "source_sha256": sources,
        "cache_manifest_sha256": manifest_sha,
        "data_usage": {
            "fit_split": "train",
            "validation_or_test_access": "none",
            "selection": "evenly spaced sorted sequences; midpoint usable cache pair",
            "samples": inputs,
            "full_shard_bytes_verified": False,
        },
        "runtime": {"device": str(device), "seconds": time.monotonic() - started},
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "Small train diagnostic, not independent physical correspondences.",
            "Numerical inverse cycle gains are by construction, not accuracy gains.",
            "Affine fitted to predicted fields without annotation conditioning.",
            "Rejected affine candidates cannot qualify via partial sample coverage.",
        ],
        "variants": {
            name: {
                "coverage_complete": len(items) == len(selected),
                "summary": summary_for(items),
                "rows": items,
            }
            for name, items in rows.items()
        },
        "diagnostics": diagnostics,
    }
    report["report_sha256"] = canonical_hash(report)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for name, item in report["variants"].items():
        stats = item["summary"]
        print(
            name,
            json.dumps(
                {
                    "complete": item["coverage_complete"],
                    "joint_pass": stats["joint_frame_pass_rate"],
                    **{
                        d: {
                            k: stats["directions"][d][k]["median"]
                            for k in ("bbox_iou", "cycle_p95_pixels", "roi_cycle_p95_pixels")
                        }
                        for d in DIRECTIONS
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
