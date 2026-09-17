#!/usr/bin/env python3
"""Re-evaluate existing complete checkpoints with corrected v4 coordinates.

This is an engineering audit, NOT an automatic generator-training authorization.
Both native point-map directions are named explicitly to avoid warp-name ambiguity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from aero_ir.data.antiuav import load_split_manifest
from aero_ir.registration.geometry import CONVENTION, from_superfusion
from aero_ir.registration.qualification_v4 import (
    THRESHOLDS,
    direction_statistics,
    gate_report,
    split_report,
)
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import (
    _balanced_indices,
    _batches,
    _candidate_pairs,
    _iter_pairs,
)


def checkpoint_provenance(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("aero_registration", {})
    if (
        metadata.get("dataset") != "Anti-UAV300"
        or metadata.get("fit_split") != "train"
        or metadata.get("validation_or_test_access") != "none"
        or metadata.get("epochs") != 300
        or payload.get("epoch") != 300
        or not isinstance(payload.get("DM"), dict)
    ):
        raise ValueError("audit requires an actually completed 300-epoch train-only checkpoint")
    return {"sha256": file_sha256(path), "epoch": payload["epoch"], "training_metadata": metadata}


def expected_pairs(root: Path, samples: int) -> tuple[dict[str, set[tuple[str, int]]], dict]:
    expected = {}
    inputs = {}
    for split in ("train", "val"):
        expected[split] = set()
        manifest = root / "label_new" / f"{split}.json"
        inputs[str(manifest.relative_to(root))] = file_sha256(manifest)
        for sequence in sorted(load_split_manifest(root, split)):
            directory = root / split / sequence
            indices, _, _ = _candidate_pairs(directory)
            for name in ("visible.json", "infrared.json"):
                path = directory / name
                inputs[str(path.relative_to(root))] = file_sha256(path)
            expected[split].update(
                (sequence, indices[int(i)]) for i in _balanced_indices(len(indices), samples)
            )
    return expected, inputs


def source_hashes(project: Path) -> dict:
    paths = [
        project / "scripts/audit_antiuav300_registration_v4.py",
        project / "scripts/audit_antiuav300_dense_registration.py",
        project / "src/aero_ir/data/antiuav.py",
    ]
    paths.extend(sorted((project / "src/aero_ir/registration").glob("*.py")))
    return {str(path.relative_to(project)): file_sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.samples_per_sequence < 0 or args.batch_size <= 0:
        parser.error("samples must be nonnegative; batch size must be positive")
    if args.out.exists():
        parser.error("refusing to overwrite an existing audit; choose a new --out")
    root = args.root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no training or audit was started")
    provenance = checkpoint_provenance(args.checkpoint)
    expected, inputs = expected_pairs(root, args.samples_per_sequence)
    project = Path(__file__).resolve().parents[1]
    code_before = source_hashes(project)
    model = load_superfusion_matcher(args.checkpoint, device).eval()
    rows = defaultdict(list)
    observed = {split: set() for split in expected}
    with torch.inference_mode():
        for batch in _batches(
            _iter_pairs(root, ("train", "val"), args.samples_per_sequence), args.batch_size
        ):

            def images(name, batch=batch):
                return (
                    torch.from_numpy(np.stack([getattr(pair, name) for pair in batch]))
                    .permute(0, 3, 1, 2)
                    .to(device=device, dtype=torch.float32)
                    / 255
                )

            visible, infrared = images("visible"), images("infrared")
            visible_boxes = torch.as_tensor(np.stack([p.source_box for p in batch]), device=device)
            infrared_boxes = torch.as_tensor(np.stack([p.target_box for p in batch]), device=device)
            forward = from_superfusion(model(infrared, visible, direction="visible_to_infrared"))
            reverse = from_superfusion(model(infrared, visible, direction="infrared_to_visible"))
            f = direction_statistics(forward, reverse, infrared_boxes, visible_boxes)
            r = direction_statistics(reverse, forward, visible_boxes, infrared_boxes)
            for pair, forward_row, reverse_row in zip(batch, f, r, strict=True):
                key = (pair.sequence_id, pair.frame_index)
                if key not in expected[pair.split] or key in observed[pair.split]:
                    raise ValueError("unexpected or duplicate evaluated frame")
                observed[pair.split].add(key)
                rows[pair.split].append(
                    {
                        "sequence_id": pair.sequence_id,
                        "frame_index": pair.frame_index,
                        "ir_to_rgb_points": forward_row,
                        "rgb_to_ir_points": reverse_row,
                    }
                )
            count = sum(len(items) for items in rows.values())
            if count % (args.batch_size * 20) == 0:
                print(f"v4: audited {count} paired frames", flush=True)
    if (
        source_hashes(project) != code_before
        or file_sha256(args.checkpoint) != provenance["sha256"]
    ):
        raise RuntimeError("code or checkpoint changed during audit; refusing mixed provenance")
    if any(file_sha256(root / name) != digest for name, digest in inputs.items()):
        raise RuntimeError("annotations changed during audit")
    metrics = {split: split_report(rows[split]) for split in ("train", "val")}
    report = {
        "schema_version": 4,
        "kind": "antiuav300_coordinate_consistent_geometry_audit_v4",
        "coordinate_convention": CONVENTION,
        "raw_field_conversion": "endpoint_grid + raw - pixel_centre_grid; exactly once",
        "root": str(root),
        "checkpoint": provenance,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        "source_sha256": code_before,
        "annotation_sha256": inputs,
        "data_usage": {
            "splits": ["train", "val"],
            "test_access": "none",
            "samples_per_sequence": args.samples_per_sequence,
            "exact_pair_coverage": observed == expected,
        },
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "gates": gate_report(
            metrics,
            exhaustive=args.samples_per_sequence == 0,
            complete_coverage=observed == expected,
        ),
        "limitations": [
            "boxes, local topology and cycles are engineering proxies, not dense ground truth",
            "independently reviewed correspondence/occlusion evidence is not yet available",
            "annotation hashes bind frame selection; video bytes are not hashed by this audit",
            "v2/v3 scores are not directly comparable to corrected v4 measurements",
        ],
        "rows": dict(rows),
    }
    report["audit_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"metrics": metrics, "gates": report["gates"]}, indent=2))
    print(f"wrote {args.out}; generator training remains HOLD")
    # A successfully written diagnostic is not a qualification pass.
    raise SystemExit(3)


if __name__ == "__main__":
    main()
