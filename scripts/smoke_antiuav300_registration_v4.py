#!/usr/bin/env python3
"""Bounded, train-only GPU check of the actual v4 FP32 optimizer path; no weight saves."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from aero_ir.data.antiuav import load_split_manifest
from aero_ir.registration.geometry import CONVENTION
from aero_ir.registration.protocol_v4 import (
    BATCH_SIZE,
    EPOCHS,
    INITIAL_CHECKPOINT_SHA256,
    LEARNING_RATE,
    PAIRS_PER_SEQUENCE_PER_EPOCH,
)
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.registration.training_v4 import training_step
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import _iter_pairs
from scripts.audit_antiuav300_registration_v4 import checkpoint_provenance, source_hashes


def load_smoke_pairs(root: Path, count: int):
    """Read one usable pair per sequence, train only; never build the full cache."""
    generator = _iter_pairs(root, ("train",), 1)
    try:
        pairs = list(itertools.islice(generator, count))
    finally:
        generator.close()  # close the last video capture even when islice stops early
    if len(pairs) != count or any(p.split != "train" for p in pairs):
        raise ValueError(f"expected {count} train-only pairs, got {len(pairs)}")
    if len({(p.sequence_id, p.frame_index) for p in pairs}) != count:
        raise ValueError("duplicate smoke pair")
    return pairs


def _code_hashes(project):
    result = source_hashes(project)
    for name in (
        "scripts/smoke_antiuav300_registration_v4.py",
        "scripts/run_antiuav300_registration_v4_smoke.sh",
        "scripts/train_antiuav300_registration_v4.py",
    ):
        result[name] = file_sha256(project / name)
    return result


def run_smoke(args, result: dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; smoke requires an allocated working GPU")
    device = torch.device("cuda")
    if version("kornia") != "0.6.5":
        raise RuntimeError("registration smoke requires pinned kornia==0.6.5")
    digest = file_sha256(args.checkpoint)
    if digest != INITIAL_CHECKPOINT_SHA256:
        raise ValueError("smoke must initialize from the same frozen v2 checkpoint as v4 training")
    provenance = checkpoint_provenance(args.checkpoint)
    project = Path(__file__).resolve().parents[1]
    before_code = _code_hashes(project)
    result.update(
        {
            "initial_checkpoint": provenance,
            "source_sha256": before_code,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project, text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project, text=True
            ).splitlines(),
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "kornia": version("kornia"),
                "numpy": np.__version__,
                "device": torch.cuda.get_device_name(device),
                "device_total_memory_mib": torch.cuda.get_device_properties(device).total_memory
                / 2**20,
            },
        }
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = load_split_manifest(args.root, "train")
    if len(manifest) != 160:
        raise ValueError("expected official 160-sequence training split")
    count = args.steps * BATCH_SIZE
    print(
        f"Decoding {count} train pairs (one per sequence); no validation/test access...", flush=True
    )
    inputs = {"label_new/train.json": file_sha256(args.root / "label_new/train.json")}
    # Freeze selected annotation files before decoding; generator follows sorted manifest order.
    # Bind eligibility/selection even for sequences skipped for having no usable pair.
    for sequence in sorted(manifest):
        for name in ("visible.json", "infrared.json"):
            relative = f"train/{sequence}/{name}"
            inputs[relative] = file_sha256(args.root / relative)
    pairs = load_smoke_pairs(args.root, count)
    result["annotation_sha256"] = inputs
    result["pairs"] = [
        {
            "sequence_id": p.sequence_id,
            "frame_index": p.frame_index,
            "decoded_visible_sha256": hashlib.sha256(p.visible.tobytes()).hexdigest(),
            "decoded_infrared_sha256": hashlib.sha256(p.infrared.tobytes()).hexdigest(),
            "visible_box": p.source_box.tolist(),
            "infrared_box": p.target_box.tolist(),
        }
        for p in pairs
    ]
    model = load_superfusion_matcher(args.checkpoint, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    total_steps = EPOCHS * len(manifest) * PAIRS_PER_SEQUENCE_PER_EPOCH // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    result["full_schedule_optimizer_steps"] = total_steps
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        batch = pairs[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        visible = torch.from_numpy(np.stack([p.visible for p in batch])).permute(0, 3, 1, 2)
        infrared = torch.from_numpy(np.stack([p.infrared for p in batch])).permute(0, 3, 1, 2)
        visible = visible.to(device=device, dtype=torch.float32) / 255
        infrared = infrared.to(device=device, dtype=torch.float32) / 255
        vb = torch.as_tensor(
            np.stack([p.source_box for p in batch]), device=device, dtype=torch.float32
        )
        ib = torch.as_tensor(
            np.stack([p.target_box for p in batch]), device=device, dtype=torch.float32
        )
        lr_used = optimizer.param_groups[0]["lr"]
        values, checks = training_step(
            model, optimizer, visible, infrared, vb, ib, diagnostics=True
        )
        scheduler.step()
        torch.cuda.synchronize(device)
        record = {
            "step": step + 1,
            "loss": values,
            "checks": checks,
            "learning_rate_used": lr_used,
            "elapsed_seconds": time.perf_counter() - started,
        }
        result["steps"].append(record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    result["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    if file_sha256(args.checkpoint) != digest or _code_hashes(project) != before_code:
        raise RuntimeError("checkpoint or implementation changed during smoke")
    if any(file_sha256(args.root / name) != value for name, value in inputs.items()):
        raise RuntimeError("train annotations changed during smoke")
    result.update({"ok": True, "engineering_smoke": "pass", "initial_checkpoint_unchanged": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    args.root, args.checkpoint = args.root.resolve(), args.checkpoint.resolve()
    if not args.root.is_dir() or not args.checkpoint.is_file():
        parser.error("dataset root and frozen initial checkpoint must exist")
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a fresh smoke output path")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "kind": "antiuav300_registration_v4_gpu_engineering_smoke",
        "ok": False,
        "engineering_smoke": "fail",
        "requested_steps": args.steps,
        "batch_size": BATCH_SIZE,
        "precision": "float32",
        "coordinate_convention": CONVENTION,
        "data_usage": "train only, first usable pair of sorted training sequences",
        "root": str(args.root),
        "seed": 0,
        "steps": [],
        "checkpoint_saved": False,
        "generator_training_eligible": "hold",
        "limitations": [
            "not registration accuracy, convergence or full-training qualification",
            "timing includes diagnostics; first step includes cold GPU initialization",
            "temporary parameter snapshots increase peak memory",
            "fixed seed is not a promise of bitwise CUDA grid-sample backward replay",
        ],
    }
    try:
        run_smoke(args, result)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    result["report_sha256"] = canonical_hash(result)
    path = args.output_dir / "smoke_report.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"v4 engineering smoke: {'PASS' if result['ok'] else 'FAIL'}; wrote {path}", flush=True)
    if not result["ok"]:
        print(json.dumps(result.get("error")), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
