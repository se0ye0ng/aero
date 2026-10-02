"""Evaluate the ONE shared fit-capacity head on train panels; no case routing."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from aero_ir.registration.residual_velocity import ResidualVelocityHead
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_registration_residual_pilot import pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.probe_registration_balanced_scale_learning import panels, screen_panel
from scripts.probe_registration_fit_capacity import KIND, LR, STEPS
from scripts.screen_registration_scale_learning import paired_summary
from scripts.train_registration_residual_pilot import load_checkpoint


def verified_shared_head(folder, report, spec):
    if report["kind"] != KIND or spec["kind"] != KIND:
        raise ValueError("not a fit-capacity diagnostic")
    if (
        report["spec_sha256"] != canonical_hash(spec)
        or spec["steps_per_head"] != STEPS
        or spec["lr"] != LR
        or spec["loss"] != "unchanged_v7_geometry"
        or report["source_sha256"] != spec["source_sha256"]
    ):
        raise ValueError("diagnostic specification differs")
    for name, digest in spec["source_sha256"].items():
        if file_sha256(name) != digest or file_sha256(folder / "sources" / name) != digest:
            raise ValueError(f"source drift: {name}")
    shared = report["outcomes"]["shared"]
    path = folder / "shared_diagnostic_head.pth"
    if file_sha256(path) != shared["checkpoint_sha256"]:
        raise ValueError("shared checkpoint differs")
    state = torch.load(path, weights_only=True, map_location="cpu")
    identities = [r["sequence_id"] for r in spec["panel"]]
    if (
        state["kind"] != KIND
        or state["spec_sha256"] != canonical_hash(spec)
        or state["recorded_updates"] != STEPS
        or state["fit_sequence_ids"] != identities
        or shared["fit_sequence_ids"] != identities
        or [r["update"] for r in shared["trace"]] != list(range(1, STEPS + 1))
    ):
        raise ValueError("shared head provenance differs")
    head = ResidualVelocityHead()
    head.load_state_dict(state["head_state"], strict=True)
    if any(not torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError("nonfinite shared head")
    return head.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fit-root", type=Path, default=Path("experiments/registration_fit_capacity_train4_01")
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
    files = {name: file_sha256(args.fit_root / name) for name in ("report.json", "spec.json")}
    fit_report = json.loads((args.fit_root / "report.json").read_text())
    spec = json.loads((args.fit_root / "spec.json").read_text())
    hashes = pilot_hashes(args.run_root)
    if (
        fit_report["pilot_artifacts_sha256"] != hashes
        or spec["initial_checkpoint_sha256"] != hashes["residual_head"]["final.pth"]
    ):
        raise ValueError("different completed initializer")
    compare(args.run_root, args.cache_root)
    head = verified_shared_head(args.fit_root, fit_report, spec)
    sources = dict(spec["source_sha256"])
    for name in (
        "scripts/screen_registration_fit_capacity.py",
        "scripts/probe_registration_balanced_scale_learning.py",
        "scripts/screen_registration_scale_learning.py",
    ):
        sources[name] = file_sha256(name)
    screen = json.loads((args.run_root / "residual_head" / "final_train_screen.json").read_text())
    manifest_hash = file_sha256(args.cache_root / "manifest.json")
    shards = json.loads((args.cache_root / "manifest.json").read_text())["shards"]
    midpoint, quarter = panels(screen, shards)
    fit_ids = {r["sequence_id"] for r in spec["panel"]}
    for sample in midpoint:
        sample["role"] = (
            "capacity_fit" if sample["sequence_id"] in fit_ids else "not_in_capacity_fit"
        )
    torch.set_num_threads(1)
    model, _ = load_checkpoint(args.run_root / "residual_head" / "final.pth", torch.device("cpu"))
    model.requires_grad_(False)
    heads = {"initial": model.head, "shared_capacity_fit": head}
    rows = {
        role: screen_panel(heads, samples, model, args.cache_root)
        for role, samples in (("train_midpoints", midpoint), ("train_quarters", quarter))
    }
    summary = {
        role: {
            arm: paired_summary(values["initial"], candidate) for arm, candidate in values.items()
        }
        for role, values in rows.items()
    }
    other = {
        a: [r for r in values if r["sequence_id"] not in fit_ids]
        for a, values in rows["train_midpoints"].items()
    }
    if any(file_sha256(args.fit_root / n) != h for n, h in files.items()):
        raise ValueError("fit evidence drift")
    verified_shared_head(args.fit_root, fit_report, spec)
    if (
        pilot_hashes(args.run_root) != hashes
        or file_sha256(args.cache_root / "manifest.json") != manifest_hash
    ):
        raise ValueError("pilot/cache drift")
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("source drift")
    result = {
        "kind": "shared_fit_capacity_transfer_screen_not_qualification",
        "device": "cpu",
        "parameter_updates": 0,
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "fit_input_sha256": files,
        "source_sha256": sources,
        "pilot_artifacts_sha256": hashes,
        "shared_checkpoint_sha256": fit_report["outcomes"]["shared"]["checkpoint_sha256"],
        "cpu_initial_vs_saved_gpu": paired_summary(
            screen["rows"], rows["train_midpoints"]["initial"]
        ),
        "summary": summary,
        "rows": rows,
        "other156_midpoints": {
            a: paired_summary(other["initial"], candidate) for a, candidate in other.items()
        },
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "One fixed shared head, no independent per-frame head selection.",
            "All observations are train sequences used in earlier main training.",
            "Only four midpoint observations entered the additional200updates.",
            "No physical-correspondence GT, held-out accuracy or qualification.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = args.out_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        if file_sha256(target) != digest:
            raise ValueError("source snapshot drift")
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                role: {
                    a: {k: v for k, v in s.items() if not k.endswith("ids")}
                    for a, s in values.items()
                }
                for role, values in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
