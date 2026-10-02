"""Matched v7 model-inference audit on fixed train/validation observations.

This evaluates centre-lattice shared fields directly, never a legacy DM payload.
Completion means an engineering report was written, not physical qualification.
No test frames, parameter fitting, checkpoint selection or threshold search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from aero_ir.data.antiuav import load_split_manifest
from aero_ir.registration.geometry import CONVENTION
from aero_ir.registration.qualification_v4 import (
    THRESHOLDS,
    direction_pass,
    direction_statistics,
    gate_report,
)
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import _batches, _iter_pairs
from scripts.audit_antiuav300_registration_v4 import expected_pairs
from scripts.train_antiuav300_registration_v7 import load_trained
from scripts.verify_registration_v7 import ARMS, check, compare_arms, read_json, verify_arm

SPLITS = ("train", "val")


def build_panel(root, cache_root, samples):
    """Freeze exact annotation-selected identities before any model inference."""
    check(samples in (0, 8), "use the frozen eight-frame screen or exhaustive eligible pairs")
    root = root.resolve()
    sequences = {split: set(load_split_manifest(root, split)) for split in SPLITS}
    check(all(sequences.values()), "empty official split")
    check(not sequences["train"] & sequences["val"], "train/validation sequence overlap")
    paths = [root / "label_new" / f"{split}.json" for split in SPLITS]
    for split, names in sequences.items():
        for name in names:
            directory = (root / split / name).resolve()
            check(
                directory.parent == (root / split).resolve(),
                "sequence path is not a direct child of its official split",
            )
            paths.extend(directory / f"{modality}.json" for modality in ("visible", "infrared"))
    before = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    expected, inputs = expected_pairs(root, samples)
    check(inputs == before, "annotations changed while selecting the panel")
    cache = read_json(cache_root / "manifest.json")
    check(
        cache["fit_split"] == "train" and cache["validation_or_test_access"] == "none",
        "cache is not train-only",
    )
    check(
        cache["split_manifest_sha256"] == inputs["label_new/train.json"],
        "evaluation train split differs from training cache",
    )
    check(
        {s["sequence_id"] for s in cache["shards"]} == sequences["train"],
        "training cache and official train sequences differ",
    )
    selection = {split: [list(v) for v in sorted(expected[split])] for split in SPLITS}
    missing = {
        split: sorted(sequences[split] - {sid for sid, _ in expected[split]}) for split in SPLITS
    }
    return expected, {
        "kind": "v7_frozen_annotation_selected_evaluation_panel",
        "samples_per_sequence": samples,
        "selection": selection,
        "selection_sha256": canonical_hash(selection),
        "annotation_sha256": inputs,
        "official_sequences": {k: len(v) for k, v in sequences.items()},
        "sequences_without_eligible_pairs": missing,
        "test_access": "none",
        "selection_scope": "both-present frames with valid in-bounds native boxes",
        "frame_pairing": "same frame index; no lag optimization or pairing modification",
    }


def pair_digest(pair):
    digest = hashlib.sha256()
    digest.update(json.dumps([pair.split, pair.sequence_id, pair.frame_index]).encode())
    for key in ("visible", "infrared", "source_box", "target_box"):
        value = np.ascontiguousarray(getattr(pair, key))
        digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def infer_rows(model, batch, device):
    """One shared prediction per batch; fields() already uses centre coordinates."""
    check(bool(batch), "empty inference batch")
    images = [
        torch.from_numpy(np.stack([getattr(p, key) for p in batch]))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        / 255
        for key in ("visible", "infrared")
    ]
    boxes = [
        torch.as_tensor(np.stack([getattr(p, key) for p in batch]), device=device)
        for key in ("source_box", "target_box")
    ]
    visible, infrared = images
    vb, ib = boxes
    with torch.no_grad():
        forward, reverse = model.fields(infrared, visible)
        check(
            forward.shape == reverse.shape == (len(batch), 2, *infrared.shape[-2:]),
            "unexpected centre-field shape",
        )
        f = direction_statistics(forward, reverse, ib, vb)
        r = direction_statistics(reverse, forward, vb, ib)
    return [
        {
            "split": p.split,
            "sequence_id": p.sequence_id,
            "frame_index": p.frame_index,
            "array_sha256": pair_digest(p),
            "ir_to_rgb_points": a,
            "rgb_to_ir_points": b,
        }
        for p, a, b in zip(batch, f, r, strict=True)
    ]


class Coverage:
    """Streaming equivalent of v4 split_report, without retaining dense audit rows."""

    def __init__(self, expected):
        self.expected = expected
        self.seen = {s: set() for s in expected}
        self.counts = {s: defaultdict(lambda: [0, 0]) for s in expected}
        self.data_signature = hashlib.sha256()

    def add(self, row):
        split = row["split"]
        key = (row["sequence_id"], row["frame_index"])
        check(split in self.expected, "unexpected evaluated split")
        check(key in self.expected[split], "unexpected evaluated frame")
        check(key not in self.seen[split], "duplicate evaluated frame")
        self.seen[split].add(key)
        passed = all(direction_pass(row[k]) for k in ("ir_to_rgb_points", "rgb_to_ir_points"))
        count = self.counts[split][row["sequence_id"]]
        count[0] += int(passed)
        count[1] += 1
        self.data_signature.update(
            json.dumps([split, *key, row["array_sha256"]], separators=(",", ":")).encode() + b"\n"
        )

    def finish(self):
        check(self.seen == self.expected, "incomplete evaluation frame coverage")
        metrics = {}
        for split, counts in self.counts.items():
            check(bool(counts), f"no eligible evaluation pairs in {split}")
            rates = {sid: good / total for sid, (good, total) in sorted(counts.items())}
            total = sum(n for _, n in counts.values())
            metrics[split] = {
                "evaluated_pairs": total,
                "sequences": len(counts),
                "joint_frame_pass_rate": sum(n for n, _ in counts.values()) / total,
                "sequence_macro_pass_rate": sum(rates.values()) / len(rates),
                "per_sequence_pass_rate": rates,
            }
        return metrics


def source_hashes(project):
    # The trainer is imported for its architecture-specific loader; not executed.
    paths = [
        "scripts/audit_antiuav300_registration_v7.py",
        "scripts/audit_antiuav300_registration_v4.py",
        "scripts/audit_antiuav300_dense_registration.py",
        "scripts/verify_registration_v7.py",
        "scripts/train_antiuav300_registration_v7.py",
        "src/aero_ir/data/antiuav.py",
        "src/aero_ir/utils/manifest.py",
    ]
    paths += [
        str(p.relative_to(project)) for p in (project / "src/aero_ir/registration").glob("*.py")
    ]
    return {p: file_sha256(project / p) for p in sorted(paths)}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300")
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v7_pilot_e10_seed0"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-sequence", type=int, choices=(0, 8), default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    for name in ("root", "run_root", "cache_root", "out_dir"):
        if any(c in str(getattr(args, name)) for c in ("\r", "\n")):
            parser.error(f"newline in {name}")
    if args.out_dir.exists():
        parser.error("refusing to overwrite an audit directory; choose a fresh --out-dir")

    project = Path(__file__).resolve().parents[1]
    sources = source_hashes(project)
    expected, panel = build_panel(args.root, args.cache_root, args.samples_per_sequence)
    arms = {}
    for arm in ARMS:
        try:
            arms[arm] = verify_arm(args.run_root / arm, arm, args.cache_root)
        except (ValueError, KeyError, OSError, RuntimeError, TypeError) as error:
            arms[arm] = {"status": "verification_failed", "error": str(error)}
    comparison, error = None, None
    try:
        comparison = compare_arms(arms)
    except ValueError as exc:
        error = str(exc)
    # Do not inspect validation pixels when neither arm met its train screen.
    ready = bool(comparison and comparison["candidates_for_heldout_engineering_screen"])
    blocked_reason = (
        None
        if ready
        else (error or "Neither completed arm reached both frozen 95% train-screen pass rates.")
    )
    plan = {
        "schema_version": 1,
        "kind": "v7_inference_audit_preflight",
        "ready_for_engineering_inference": ready,
        "saved_pilots": arms,
        "comparison": comparison,
        "comparison_error": error,
        "blocked_reason": blocked_reason,
        "panel_sha256": canonical_hash(panel),
        "selected_pairs": {s: len(v) for s, v in expected.items()},
        "source_sha256": sources,
        "cache_manifest_sha256": file_sha256(args.cache_root / "manifest.json"),
        "coordinate_convention": CONVENTION,
        "thresholds": THRESHOLDS,
        "generator_training_eligible": "hold_not_qualified",
        "pixel_inference_executed": False,
    }
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir / "panel.json", panel)
    write_json(args.out_dir / "preflight.json", plan)
    print(json.dumps({"ready": ready, "selected_pairs": plan["selected_pairs"]}, indent=2))
    if args.preflight_only or not ready:
        print(
            "Preflight only; no pixel inference or generator qualification. "
            + (blocked_reason or "")
        )
        return 0 if ready else 2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; preflight preserved, no inference started")
    models = {}
    for arm in ARMS:
        checkpoint = args.run_root / arm / "shared_velocity_e10.pth"
        check(
            file_sha256(checkpoint) == arms[arm]["artifacts_sha256"][checkpoint.name],
            "checkpoint changed after evidence verification",
        )
        models[arm], _ = load_trained(checkpoint, device)
    coverage = {a: Coverage(expected) for a in ARMS}
    paired_changes = {
        s: {"both_pass": 0, "mind_only_pass": 0, "geometry_only_pass": 0, "neither_pass": 0}
        for s in SPLITS
    }
    with ExitStack() as stack:
        outputs = {a: stack.enter_context((args.out_dir / f"{a}.jsonl").open("x")) for a in ARMS}
        for batch in _batches(
            _iter_pairs(args.root, SPLITS, args.samples_per_sequence), args.batch_size
        ):
            predictions = {a: infer_rows(models[a], batch, device) for a in ARMS}
            for arm, rows in predictions.items():
                for row in rows:
                    coverage[arm].add(row)
                    outputs[arm].write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                outputs[arm].flush()
            for left, right in zip(predictions[ARMS[0]], predictions[ARMS[1]], strict=True):
                check(left["array_sha256"] == right["array_sha256"], "arm observations differ")
                flags = [
                    all(direction_pass(r[k]) for k in ("ir_to_rgb_points", "rgb_to_ir_points"))
                    for r in (left, right)
                ]
                key = {
                    (True, True): "both_pass",
                    (False, True): "mind_only_pass",
                    (True, False): "geometry_only_pass",
                    (False, False): "neither_pass",
                }[tuple(flags)]
                paired_changes[left["split"]][key] += 1
            print(
                f"v7 audited {sum(len(v) for v in coverage[ARMS[0]].seen.values())} pairs",
                flush=True,
            )

    check(source_hashes(project) == sources, "audit sources changed during inference")
    check(
        file_sha256(args.cache_root / "manifest.json") == plan["cache_manifest_sha256"],
        "training cache identity changed during inference",
    )
    for name, digest in panel["annotation_sha256"].items():
        check(file_sha256(args.root / name) == digest, "annotation drift during inference")
    for arm in ARMS:
        for name, digest in arms[arm]["artifacts_sha256"].items():
            check(file_sha256(args.run_root / arm / name) == digest, "pilot artifact drift")
    metrics = {a: coverage[a].finish() for a in ARMS}
    signatures = {a: coverage[a].data_signature.hexdigest() for a in ARMS}
    check(len(set(signatures.values())) == 1, "arms consumed different evaluation data")
    complete = not any(panel["sequences_without_eligible_pairs"].values())
    report = {
        "schema_version": 1,
        "kind": "antiuav300_v7_matched_engineering_inference_audit",
        "panel_sha256": file_sha256(args.out_dir / "panel.json"),
        "preflight_sha256": file_sha256(args.out_dir / "preflight.json"),
        "row_artifacts_sha256": {a: file_sha256(args.out_dir / f"{a}.jsonl") for a in ARMS},
        "coordinate_convention": CONVENTION,
        "field_conversion": "model.fields returns centre fields; no second raw-field conversion",
        "metrics": metrics,
        "paired_outcomes": paired_changes,
        "consumed_pair_signatures": signatures,
        "exact_eligible_pair_coverage": True,
        "all_official_sequences_have_observations": complete,
        "gates": {
            a: gate_report(
                metrics[a], exhaustive=args.samples_per_sequence == 0, complete_coverage=complete
            )
            for a in ARMS
        },
        "generator_training_eligible": "hold_not_qualified",
        "source_sha256": sources,
        "runtime": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "batch_size": args.batch_size,
        },
        "limitations": [
            "This is a ten-epoch pilot evaluation, not a completed 300-epoch final experiment.",
            "Validation was used in earlier development diagnostics; it is not an untouched test.",
            "No test access; absent/invalid-box frames are outside this eligible-pair audit.",
            "Array hashes bind decoded resized inputs, not whole source video files.",
            "Boxes, cycles and topology do not establish independent physical correspondence.",
        ],
    }
    write_json(args.out_dir / "report.json", report)
    print(
        json.dumps(
            {"report": str(args.out_dir / "report.json"), "gates": report["gates"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
