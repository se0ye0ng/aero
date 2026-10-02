#!/usr/bin/env python3
"""Compare completed registration pilots on identical decoded train/val frames.

This diagnostic accepts completed short pilots without changing the v4 qualification
protocol. Exit zero means the comparison completed, never generator authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from aero_ir.registration.geometry import CONVENTION, from_superfusion
from aero_ir.registration.qualification_v4 import (
    THRESHOLDS,
    direction_pass,
    direction_statistics,
    gate_report,
    split_report,
)
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import _batches, _iter_pairs
from scripts.audit_antiuav300_registration_v4 import expected_pairs, source_hashes

DIRECTIONS = ("ir_to_rgb_points", "rgb_to_ir_points")
MEASUREMENTS = (
    "bbox_iou",
    "centroid_shift_fraction",
    "absolute_area_ratio_change",
    "cycle_p95_pixels",
    "roi_cycle_p95_pixels",
    "roi_cycle_max_pixels",
    "valid_fraction",
    "cycle_valid_fraction",
    "roi_valid_fraction",
    "positive_jacobian_fraction",
    "roi_positive_jacobian_fraction",
    "global_nonpositive_jacobian_fraction",
)


class NumericalInverseMatcher(torch.nn.Module):
    """Preserve native IR->RGB map, derive its reciprocal without reading boxes."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, infrared, visible, *, direction):
        from scripts.probe_registration_inverse import invert_field

        raw = self.model(infrared, visible, direction="visible_to_infrared")
        if direction == "visible_to_infrared":
            return raw
        if direction != "infrared_to_visible":
            raise ValueError("unknown point-map direction")
        seed = self.model(infrared, visible, direction="infrared_to_visible")
        inverse, _ = invert_field(from_superfusion(raw), from_superfusion(seed), iterations=20)
        return inverse - from_superfusion(torch.zeros_like(inverse))


def checkpoint_info(path: Path) -> dict:
    before = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload.get("aero_registration", {})
    epoch = payload.get("epoch")
    if (
        meta.get("dataset") != "Anti-UAV300"
        or meta.get("fit_split") != "train"
        or meta.get("validation_or_test_access") != "none"
        or not isinstance(epoch, int)
        or epoch <= 0
        or epoch != meta.get("epochs")
        or not isinstance(payload.get("DM"), dict)
        or not payload["DM"]
    ):
        raise ValueError(f"requires a completed train-only checkpoint: {path}")
    if any(not torch.isfinite(v).all() for v in payload["DM"].values()):
        raise ValueError(f"nonfinite checkpoint tensors: {path}")
    if before != file_sha256(path):
        raise RuntimeError(f"checkpoint changed while loading: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": before,
        "epoch": epoch,
        "training_metadata": meta,
    }


def summarize(values) -> dict:
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "missing_or_nonfinite": len(values) - len(finite),
        "mean": float(np.mean(finite)) if finite else None,
        "median": float(np.median(finite)) if finite else None,
        "p95_across_frames": float(np.quantile(finite, 0.95)) if finite else None,
    }


def summary_for(rows: list[dict]) -> dict:
    result = split_report(rows)
    result["directions"] = {}
    for direction in DIRECTIONS:
        items = [r[direction] for r in rows]
        result["directions"][direction] = {
            "direction_pass_rate": sum(direction_pass(r) for r in items) / len(items),
            **{key: summarize([r.get(key) for r in items]) for key in MEASUREMENTS},
        }
    return result


def paired_deltas(rows, reference) -> dict:
    def index(items):
        keys = [(r["sequence_id"], r["frame_index"]) for r in items]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate frame key")
        return dict(zip(keys, items, strict=True))

    actual, base = index(rows), index(reference)
    if actual.keys() != base.keys():
        raise ValueError("cannot compare different frame selections")
    result = {}
    for direction in DIRECTIONS:
        result[direction] = {}
        for metric in MEASUREMENTS:
            deltas = []
            for key in base:
                a, b = actual[key][direction].get(metric), base[key][direction].get(metric)
                valid = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
                deltas.append(a - b if valid else None)
            result[direction][metric] = summarize(deltas)
    return result


def evaluate(models, batches, expected, log_every=10):
    rows = {name: {split: [] for split in expected} for name in models}
    observed = {split: set() for split in expected}
    decoded = hashlib.sha256()
    start = time.monotonic()
    with torch.inference_mode():
        for batch_id, batch in enumerate(batches, 1):
            for pair in batch:
                key = (pair.sequence_id, pair.frame_index)
                if key not in expected[pair.split] or key in observed[pair.split]:
                    raise ValueError("unexpected or duplicate evaluated frame")
                observed[pair.split].add(key)
                decoded.update(canonical_hash([pair.split, *key]).encode())
                for value in (pair.visible, pair.infrared, pair.source_box, pair.target_box):
                    array = np.ascontiguousarray(value)
                    decoded.update(canonical_hash([str(array.dtype), list(array.shape)]).encode())
                    decoded.update(array.tobytes())
            for name, model in models.items():
                device = next(model.parameters()).device

                def images(attr, batch=batch, device=device):
                    array = np.stack([getattr(pair, attr) for pair in batch])
                    return torch.from_numpy(array).permute(0, 3, 1, 2).to(device).float() / 255

                visible, infrared = images("visible"), images("infrared")
                vb = torch.as_tensor(np.stack([p.source_box for p in batch]), device=device)
                ib = torch.as_tensor(np.stack([p.target_box for p in batch]), device=device)
                forward = from_superfusion(
                    model(infrared, visible, direction="visible_to_infrared")
                )
                reverse = from_superfusion(
                    model(infrared, visible, direction="infrared_to_visible")
                )
                f = direction_statistics(forward, reverse, ib, vb)
                r = direction_statistics(reverse, forward, vb, ib)
                for pair, first, second in zip(batch, f, r, strict=True):
                    for item in (first, second):
                        item["global_nonpositive_jacobian_fraction"] = (
                            1 - item["positive_jacobian_fraction"] if item["finite_field"] else None
                        )
                    rows[name][pair.split].append(
                        {
                            "sequence_id": pair.sequence_id,
                            "frame_index": pair.frame_index,
                            DIRECTIONS[0]: first,
                            DIRECTIONS[1]: second,
                        }
                    )
            if batch_id == 1 or batch_id % log_every == 0:
                count = sum(len(v) for v in observed.values())
                print(
                    f"evaluated {count} identical pairs for {len(models)} models; "
                    f"elapsed {time.monotonic() - start:.1f}s",
                    flush=True,
                )
    if observed != expected:
        raise ValueError("incomplete frame coverage; refusing comparison")
    return rows, decoded.hexdigest()


def write_tables(out: Path, report: dict) -> None:
    table = []
    for name, result in report["models"].items():
        for split, metrics in result["metrics"].items():
            for direction, stats in metrics["directions"].items():
                record = {
                    "model": name,
                    "split": split,
                    "direction": direction,
                    "frames": metrics["evaluated_pairs"],
                    "joint_frame_pass_rate": metrics["joint_frame_pass_rate"],
                    "sequence_macro_pass_rate": metrics["sequence_macro_pass_rate"],
                    "direction_pass_rate": stats["direction_pass_rate"],
                }
                for metric in MEASUREMENTS:
                    for stat in ("mean", "median", "missing_or_nonfinite"):
                        record[f"{metric}_{stat}"] = stats[metric][stat]
                table.append(record)
    with (out / "summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "# Registration pilot comparison",
        "",
        "Sampled engineering diagnostic; generator eligibility remains HOLD.",
        "",
        "Cycle columns are medians across frames of within-frame p95 errors, in 256x256 "
        "network pixels. They are not native camera pixels or training losses. "
        "Invalid measurements are counted in summary.csv/report.json and fail the gate.",
        "",
        "| Model | Split | Point-map direction | Mean IoU | Cycle p95 median (px) | "
        "ROI p95 median (px) | Mean nonpositive Jacobian fraction | Joint pass |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(v):
        return "missing" if v is None else f"{v:.4f}"

    for row in table:
        vals = [row[k] for k in ("model", "split", "direction")]
        vals += [
            fmt(row[k])
            for k in (
                "bbox_iou_mean",
                "cycle_p95_pixels_median",
                "roi_cycle_p95_pixels_median",
                "global_nonpositive_jacobian_fraction_mean",
                "joint_frame_pass_rate",
            )
        ]
        lines.append("| " + " | ".join(vals) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--baseline", default="v4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--inverse-ir-to-rgb",
        action="append",
        default=[],
        metavar="NAME",
        help="derive reciprocal of named model's native IR->RGB point map",
    )
    args = parser.parse_args()
    if min(args.samples_per_sequence, args.batch_size, args.log_every) <= 0:
        parser.error("samples, batch size and log interval must be positive")
    paths = {}
    for spec in args.checkpoint:
        name, sep, value = spec.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in paths:
            parser.error("checkpoint requires a unique NAME=PATH")
        paths[name] = Path(value).resolve()
    if args.baseline not in paths:
        parser.error("baseline must be included among checkpoints")
    if any(name not in paths or name == args.baseline for name in args.inverse_ir_to_rgb):
        parser.error("inverse adapters require named non-baseline checkpoints")
    for path in [args.root, args.output_dir, *paths.values()]:
        if any(char in str(path) for char in "\r\n"):
            parser.error("a path contains a newline")
    if args.output_dir.exists():
        parser.error("output directory exists; choose a fresh comparison run ID")
    root = args.root.resolve()
    device = torch.device(args.device)
    if not args.preflight_only and device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    provenance = {name: checkpoint_info(path) for name, path in paths.items()}
    baseline = provenance[args.baseline]
    cache_hash = baseline["training_metadata"].get("cache_manifest_sha256")
    if not cache_hash:
        parser.error("baseline has no cache hash")
    for name, info in provenance.items():
        meta = info["training_metadata"]
        if meta.get("cache_manifest_sha256") != cache_hash:
            parser.error(f"{name}: training cache differs from baseline")
        if name != args.baseline and meta.get("initial_checkpoint_sha256") != baseline["sha256"]:
            parser.error(f"{name}: not initialized from the supplied baseline")
    expected, annotations = expected_pairs(root, args.samples_per_sequence)
    if any(not keys for keys in expected.values()):
        parser.error("empty evaluation split")
    print(
        json.dumps(
            {
                "checkpoints": {n: p["epoch"] for n, p in provenance.items()},
                "expected_pairs": {s: len(v) for s, v in expected.items()},
            },
            indent=2,
        ),
        flush=True,
    )
    if args.preflight_only:
        print("Preflight passed; no inference performed and no output directory created.")
        return
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    project = Path(__file__).resolve().parents[1]

    def code_hashes():
        return {
            **source_hashes(project),
            str(Path(__file__).relative_to(project)): file_sha256(__file__),
            "src/aero_ir/utils/manifest.py": file_sha256(project / "src/aero_ir/utils/manifest.py"),
            "scripts/probe_registration_inverse.py": file_sha256(
                project / "scripts/probe_registration_inverse.py"
            ),
        }

    code_before = code_hashes()
    models = {name: load_superfusion_matcher(path, device).eval() for name, path in paths.items()}
    for name in args.inverse_ir_to_rgb:
        models[name] = NumericalInverseMatcher(models[name]).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, decoded_hash = evaluate(
        models,
        _batches(_iter_pairs(root, ("train", "val"), args.samples_per_sequence), args.batch_size),
        expected,
        args.log_every,
    )
    if code_hashes() != code_before:
        raise RuntimeError("source changed during comparison")
    if any(file_sha256(paths[n]) != p["sha256"] for n, p in provenance.items()):
        raise RuntimeError("checkpoint changed during comparison")
    if any(file_sha256(root / p) != h for p, h in annotations.items()):
        raise RuntimeError("annotations changed during comparison")
    report = {
        "schema_version": 1,
        "kind": "antiuav300_paired_checkpoint_diagnostic",
        "baseline": args.baseline,
        "field_adapters": {
            name: (
                "native_ir_to_rgb_plus_newton_inverse_20_iterations"
                if name in args.inverse_ir_to_rgb
                else "original_learned_pair"
            )
            for name in models
        },
        "coordinate_convention": CONVENTION,
        "thresholds": THRESHOLDS,
        "source_sha256": code_before,
        "annotation_sha256": annotations,
        "data_usage": {
            "splits": ["train", "val"],
            "test_access": "none",
            "samples_per_sequence": args.samples_per_sequence,
            "exact_pair_coverage": True,
            "decoded_inputs_sha256": decoded_hash,
            "selection": "v4 endpoint-inclusive usable pairs per sequence",
        },
        "runtime": {
            "device": str(device),
            "torch": str(torch.__version__),
            "elapsed_seconds": time.monotonic() - started,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "limitations": [
            "Sampled developmental train/val diagnostic, not exhaustive or independent evidence.",
            "Cycle pixels refer to 256x256 network images; no native-camera accuracy claim.",
            "Quantiles summarize finite measurements; missing counts are explicit and fail gates.",
            "Paired deltas are candidate minus baseline on the identical frame and direction.",
            "Constructed inverse cycle gains are algebraic, not physical correspondence evidence.",
        ],
        "models": {},
    }
    for name in models:
        metrics = {s: summary_for(rows[name][s]) for s in expected}
        report["models"][name] = {
            "checkpoint": provenance[name],
            "metrics": metrics,
            "gates": gate_report(metrics, exhaustive=False, complete_coverage=True),
            "paired_deltas_vs_baseline": {
                s: paired_deltas(rows[name][s], rows[args.baseline][s]) for s in expected
            },
            "rows": rows[name],
        }
    report["report_sha256"] = canonical_hash(report)
    with (args.output_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    write_tables(args.output_dir, report)
    print(f"Comparison complete: {args.output_dir / 'summary.md'}; generator remains HOLD.")


if __name__ == "__main__":
    main()
