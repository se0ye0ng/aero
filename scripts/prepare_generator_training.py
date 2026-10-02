#!/usr/bin/env python3
"""Prepare a qualification-gated generator experiment.

This is deliberately an orchestration layer, not a pretend diffusion trainer.  The
repository does not vendor DiffV2IR weights or a diffusion implementation.  A real
generator is therefore supplied through an external command after registration
qualification.  The script freezes the inputs, resolves the real/generated mix, and
records the exact command and hashes before launching it.

The implemented evidence reader accepts the existing v4 engineering audit only:
its gates are recomputed and remain HOLD without physical correspondence evidence.
There is currently no implemented scientific qualification/export authorizer.
A manually supplied ``pass`` string must never turn this planner into one.
``--plan-only`` records a blocked plan without launching a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aero_ir.data.mixing import resolve_mix
from aero_ir.registration.geometry import CONVENTION
from aero_ir.registration.qualification_v4 import THRESHOLDS, gate_report, split_report
from aero_ir.utils.manifest import canonical_hash, file_sha256


def _json(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()

    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    value = json.loads(data, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def validate_evidence(report: dict, pair_manifest: dict) -> dict:
    """Recompute existing engineering evidence; never invent a physical-pass rule.

    Shared train-list identity is checked, not a claim that the raw cache is a
    registered export. No checkpoint inference or cached-array replay takes place.
    Unknown qualification protocols need their own implemented evidence validator.
    """
    if (report.get("kind") != "antiuav300_coordinate_consistent_geometry_audit_v4"
            or report.get("schema_version") != 4):
        raise ValueError("unsupported qualification protocol; a pass string is not evidence")
    if canonical_hash({k: v for k, v in report.items() if k != "audit_sha256"}) != report.get(
        "audit_sha256"
    ):
        raise ValueError("registration report content hash mismatch")
    if report.get("coordinate_convention") != CONVENTION or report.get("thresholds") != THRESHOLDS:
        raise ValueError("coordinate convention or qualification thresholds differ")
    usage = report.get("data_usage", {})
    if (set(usage.get("splits", [])) != {"train", "val"}
            or usage.get("test_access") != "none"
            or type(usage.get("exact_pair_coverage")) is not bool
            or type(usage.get("samples_per_sequence")) is not int
            or usage["samples_per_sequence"] < 0):
        raise ValueError("invalid engineering audit split/coverage metadata")
    rows = report.get("rows", {})
    if set(rows) != {"train", "val"}:
        raise ValueError("audit must retain train and validation measurements")
    sequence_sets = {}
    for split, items in rows.items():
        if not items:
            raise ValueError("empty audit split")
        keys = [(r["sequence_id"], r["frame_index"]) for r in items]
        if (len(keys) != len(set(keys)) or any(
                not isinstance(s, str) or not s or type(i) is not int or i < 0 for s, i in keys)):
            raise ValueError("invalid/duplicate frame identities")
        sequence_sets[split] = {s for s, _ in keys}
    if sequence_sets["train"] & sequence_sets["val"]:
        raise ValueError("train/validation sequences overlap")
    metrics = {split: split_report(items) for split, items in rows.items()}
    if metrics != report.get("metrics"):
        raise ValueError("reported metrics differ from per-frame measurements")
    gates = gate_report(metrics, exhaustive=usage["samples_per_sequence"] == 0,
                        complete_coverage=usage["exact_pair_coverage"])
    if gates != report.get("gates"):
        raise ValueError("reported gates differ from recomputed evidence; no manual pass override")
    if (pair_manifest.get("kind") != "antiuav300_registration_v2_full_train_cache"
            or pair_manifest.get("schema_version") != 2
            or pair_manifest.get("fit_split") != "train"
            or pair_manifest.get("validation_or_test_access") != "none"):
        raise ValueError("explicit supported train-only pair cache required")
    if canonical_hash({k: v for k, v in pair_manifest.items() if k != "cache_manifest_sha256"}) != (
        pair_manifest.get("cache_manifest_sha256")
    ):
        raise ValueError("pair cache content hash mismatch")
    if pair_manifest.get("split_manifest_sha256") != report.get("annotation_sha256", {}).get(
        "label_new/train.json"
    ) or not pair_manifest.get("split_manifest_sha256"):
        raise ValueError("pair cache and audit use different official train lists")
    shards = pair_manifest.get("shards", [])
    if (not shards or len(shards) != pair_manifest.get("sequences")
            or any(type(s.get("pairs")) is not int or s["pairs"] < 1 for s in shards)
            or sum(s["pairs"] for s in shards) != pair_manifest.get("pairs")
            or len({s["sequence_id"] for s in shards}) != len(shards)
            or {s["sequence_id"] for s in shards} != sequence_sets["train"]):
        raise ValueError("pair cache sequences/counts do not match training audit")
    return {"status": "internally_consistent_engineering_evidence_not_qualification",
            "registration_gate": gates["generator_training_eligible"], "gates": gates,
            "scientific_authorization_verified": False,
            "registered_conditioning_export_verified": False,
            "checkpoint_inference_replayed": False, "cache_arrays_rehashed": False,
            "reasons": ["independent physical correspondence is not qualified",
                        "raw registration cache is not a qualified conditioning export"]}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator", choices=("diffv2ir", "synthetic_baseline"), required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="generator checkpoint (required for diffv2ir)"
    )
    parser.add_argument("--real-count", type=int, required=True)
    parser.add_argument("--generated-count", type=int, required=True)
    parser.add_argument("--gen-ratio", type=float, default=0.5)
    parser.add_argument("--budget-mode", choices=("fixed_total", "additive"), default="fixed_total")
    parser.add_argument("--total-images", type=int)
    parser.add_argument(
        "--command",
        help="optional external generator command; use {plan} and {output_dir} placeholders",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    report, report_digest = _json(args.qualification_report)
    pair_manifest, pair_digest = _json(args.pair_manifest)
    evidence = validate_evidence(report, pair_manifest)
    gate = evidence["registration_gate"]
    if min(args.real_count, args.generated_count) < 0:
        raise ValueError("image counts must be non-negative")
    if args.generator == "diffv2ir" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for diffv2ir")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    mix = resolve_mix(
        args.real_count,
        args.generated_count,
        args.gen_ratio,
        args.budget_mode,
        args.total_images,
    )
    command = args.command or ""
    plan = {
        "schema_version": 1,
        "kind": "aero_generator_training_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration_gate": gate,
        "evidence_validation": evidence,
        "qualification_report": {
            "path": str(args.qualification_report.resolve()),
            "sha256": report_digest,
        },
        "pair_manifest": {
            "path": str(args.pair_manifest.resolve()),
            "sha256": pair_digest,
            "content_hash": canonical_hash(pair_manifest),
        },
        "generator": {
            "name": args.generator,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        },
        "mix": {
            "n_real": mix.n_real,
            "n_generated": mix.n_gen,
            "n_total": mix.n_total,
            "realised_ratio": mix.realised_ratio,
            "budget_mode": mix.budget_mode,
            "requested_ratio": mix.gen_ratio,
        },
        "external_command": command,
        "external_command_sha256": _sha256_text(command),
        "status": "blocked_by_unqualified_evidence",
    }
    if (file_sha256(args.qualification_report) != report_digest
            or file_sha256(args.pair_manifest) != pair_digest):
        raise ValueError("input changed while preparing plan")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = args.output_dir / "generator_training_plan.json"
    with plan_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"plan": str(plan_path), "gate": gate, "status": plan["status"]}, indent=2))

    if (gate != "pass" or not evidence["scientific_authorization_verified"]
            or not evidence["registered_conditioning_export_verified"]):
        if args.plan_only:
            return 0
        raise SystemExit(
            "registration qualification is not passed; generator training is blocked "
            f"(gate={gate!r}). Use --plan-only to prepare a plan without launching."
        )
    if args.plan_only or not command:
        return 0
    rendered = [token.format(plan=str(plan_path), output_dir=str(args.output_dir))
                for token in shlex.split(command)]
    print("launching:", shlex.join(rendered))
    return subprocess.run(rendered, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
