"""CPU train160 screen of tiny scale-learning diagnostics; no fitting or GO."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from aero_ir.registration.residual_velocity import ResidualVelocityHead
from aero_ir.utils.manifest import file_sha256
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.probe_registration_scale_learning import FIT_ARMS, STEPS, cached_context
from scripts.probe_registration_v7_refinement import joint, measurements
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import load_checkpoint


def verified_heads(folder, report):
    """Bind both diagnostic heads to their recorded source, budget and base."""
    if report["kind"] != "matched_scale_consistency_cpu_learning_diagnostic":
        raise ValueError("not a scale-learning diagnostic")
    if report["settings"]["updates_per_arm"] != STEPS:
        raise ValueError("different diagnostic budget")
    for name, digest in report["source_sha256"].items():
        if file_sha256(name) != digest or file_sha256(folder / "sources" / name) != digest:
            raise ValueError(f"diagnostic source drift: {name}")
    result = {}
    for arm, weight in FIT_ARMS.items():
        path = folder / f"{arm}_diagnostic_head.pth"
        if file_sha256(path) != report["arms"][arm]["checkpoint_sha256"]:
            raise ValueError("diagnostic checkpoint hash differs")
        state = torch.load(path, weights_only=True, map_location="cpu")
        if (
            state["kind"] != "scale_loss_tiny_cpu_diagnostic_not_completed_pilot"
            or state["updates"] != STEPS
            or state["weight"] != weight
            or state["base_pilot_checkpoint_sha256"]
            != report["pilot_artifacts_sha256"]["residual_head"]["final.pth"]
            or not state["optimizer"]["state"]
            or {int(v["step"]) for v in state["optimizer"]["state"].values()} != {STEPS}
        ):
            raise ValueError("diagnostic checkpoint provenance/budget differs")
        head = ResidualVelocityHead()
        head.load_state_dict(state["head_state"], strict=True)
        if any(not torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("nonfinite head")
        result[arm] = head.eval()
    return result


def paired_summary(initial, candidate):
    if len(initial) != len(candidate) or [r["sequence_id"] for r in initial] != [
        r["sequence_id"] for r in candidate
    ]:
        raise ValueError("unmatched screen identities")
    flags = [(joint(a), joint(b)) for a, b in zip(initial, candidate, strict=True)]
    return {
        "observations": len(flags),
        "joint_passes": sum(b for _, b in flags),
        "joint_pass_rate": sum(b for _, b in flags) / len(flags),
        "gained_ids": [
            r["sequence_id"] for r, (a, b) in zip(candidate, flags, strict=True) if not a and b
        ],
        "lost_ids": [
            r["sequence_id"] for r, (a, b) in zip(candidate, flags, strict=True) if a and not b
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--diagnostic-root",
        type=Path,
        default=Path("experiments/registration_scale_learning_train4_probe4_01"),
    )
    p.add_argument(
        "--run-root", type=Path, default=Path("experiments/registration_residual_pilot_e10_seed0")
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        p.error("fresh output directory required")
    report_path = args.diagnostic_root / "report.json"
    report_hash = file_sha256(report_path)
    diagnostic = json.loads(report_path.read_text())
    hashes = pilot_hashes(args.run_root)
    if diagnostic["pilot_artifacts_sha256"] != hashes:
        raise ValueError("diagnostic used different pilot artifacts")
    compare(args.run_root, args.cache_root)
    heads = verified_heads(args.diagnostic_root, diagnostic)
    sources = {**audit_sources(), **diagnostic["source_sha256"]}
    sources["scripts/screen_registration_scale_learning.py"] = file_sha256(__file__)
    screen = json.loads((args.run_root / "residual_head" / "final_train_screen.json").read_text())
    device = torch.device("cpu")
    torch.set_num_threads(1)
    model, _ = load_checkpoint(args.run_root / "residual_head" / "final.pth", device)
    model.requires_grad_(False)
    heads = {"initial": model.head, **heads}
    fit_ids = {s["sequence_id"] for s in diagnostic["samples"] if s["role"] == "fit"}
    rows = {arm: [] for arm in heads}
    with torch.no_grad():
        for index, (saved, sample) in enumerate(
            zip(screen["rows"], screen["samples"], strict=True), 1
        ):
            arrays, digest = read_batch(args.cache_root, sample["selection"])
            if digest != sample["array_sha256"]:
                raise ValueError("screen cache observation differs")
            visible, infrared, vb, ib = tensors(arrays, device)
            context = cached_context(model, visible, infrared)
            for arm, head in heads.items():
                fields = head.fields(context["context"], context["velocity"])
                rows[arm].append(
                    {
                        "sequence_id": saved["sequence_id"],
                        "sample": sample,
                        "used_by_tiny_fit": saved["sequence_id"] in fit_ids,
                        **measurements(fields, vb, ib),
                    }
                )
            if index == 1 or index % 20 == 0:
                print(f"CPU original-image screen: {index}/{len(screen['rows'])}", flush=True)
    summary = {arm: paired_summary(rows["initial"], values) for arm, values in rows.items()}
    remaining = {
        arm: [r for r in values if not r["used_by_tiny_fit"]] for arm, values in rows.items()
    }
    other_summary = {
        arm: paired_summary(remaining["initial"], values) for arm, values in remaining.items()
    }
    cpu_vs_saved = paired_summary(screen["rows"], rows["initial"])
    if pilot_hashes(args.run_root) != hashes or file_sha256(report_path) != report_hash:
        raise ValueError("input artifact drift")
    verified_heads(args.diagnostic_root, diagnostic)
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("source drift")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = args.out_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        if file_sha256(target) != digest:
            raise ValueError("source drift during snapshot")
    report = {
        "kind": "tiny_scale_fit_full_train_screen_not_qualification",
        "device": "cpu",
        "parameter_updates": 0,
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "source_sha256": sources,
        "diagnostic_report_sha256": report_hash,
        "pilot_artifacts_sha256": hashes,
        "cpu_initial_vs_saved_gpu": cpu_vs_saved,
        "summary": summary,
        "excluding_tiny_fit_summary": other_summary,
        "rows": rows,
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "Only four train frames were used by the additional24-step fit.",
            "All160 sequences were seen by the original training pipeline.",
            "No independent physical correspondence accuracy or held-out evaluation.",
            "Not a completed full-training pilot or final300-epoch result.",
        ],
    }
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                arm: {k: v for k, v in s.items() if not k.endswith("ids")}
                for arm, s in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
