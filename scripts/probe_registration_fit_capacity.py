"""CPU fit-capacity sanity check, not registration qualification.

Fit four separate residual heads to one frame each and one shared head to all
four. Selection uses training-screen failures only. This is deliberately an
overfit diagnostic: per-frame heads are not a deployable registration method.
"""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import torch

from aero_ir.registration.residual_velocity import ResidualVelocityHead
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.analyze_registration_constraints import DIRECTIONS, GROUPS, failures
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.probe_registration_residual_learning import evaluate, fit_head
from scripts.probe_registration_scale_learning import cached_context
from scripts.probe_registration_v7_refinement import joint
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import load_checkpoint

STEPS = 200
LR = 1e-4
CASES = 4
KIND = "alignment_failure_head_fit_capacity_diagnostic"


def alignment_failure_panel(rows, count=CASES):
    """First failure per filename prefix, then first prefixes lexicographically.

    Prefix grouping avoids choosing only adjacent numbered clips. It is a naming
    heuristic, not proof of independent recordings or subjects.
    """
    if type(count) is not int or count < 1:
        raise ValueError("positive panel size required")
    if len({r["sequence_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sequence identity")
    selected, prefixes = [], set()
    for row in sorted(rows, key=lambda r: r["sequence_id"]):
        bad = set.union(*(failures(row[d]) for d in DIRECTIONS))
        prefix = row["sequence_id"].rsplit("_", 1)[0]
        if bad and bad <= set(GROUPS["alignment"]) and prefix not in prefixes:
            prefixes.add(prefix)
            selected.append(
                {
                    "sequence_id": row["sequence_id"],
                    "filename_prefix": prefix,
                    "initial_failures": sorted(bad),
                }
            )
            if len(selected) == count:
                return selected
    raise ValueError("not enough distinct-prefix alignment-only failures")


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
    hashes = pilot_hashes(args.run_root)
    compare(args.run_root, args.cache_root)
    sources = audit_sources()
    for name in (
        "scripts/probe_registration_fit_capacity.py",
        "scripts/analyze_registration_constraints.py",
        "scripts/probe_registration_residual_learning.py",
        "scripts/probe_registration_scale_learning.py",
        "scripts/probe_registration_scale_response.py",
        "scripts/probe_registration_v7_refinement.py",
        "scripts/render_registration_v7.py",
    ):
        sources[name] = file_sha256(name)
    screen = json.loads((args.run_root / "residual_head" / "final_train_screen.json").read_text())
    panel = alignment_failure_panel(screen["rows"])
    stored = {r["sequence_id"]: s for r, s in zip(screen["rows"], screen["samples"], strict=True)}
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device("cpu")
    model, _ = load_checkpoint(args.run_root / "residual_head" / "final.pth", device)
    model.requires_grad_(False)
    initial = copy.deepcopy(model.head.state_dict())
    samples = []
    for selected in panel:
        sid = selected["sequence_id"]
        arrays, digest = read_batch(args.cache_root, stored[sid]["selection"])
        if digest != stored[sid]["array_sha256"]:
            raise ValueError("selected cache bytes differ")
        visible, infrared, vb, ib = tensors(arrays, device)
        samples.append(
            {
                "sequence_id": sid,
                "role": "fit",
                "saved_stratum": "alignment_only_fail",
                "sample": stored[sid],
                "visible": visible,
                "infrared": infrared,
                "vb": vb,
                "ib": ib,
                **cached_context(model, visible, infrared),
            }
        )
    before = evaluate(model.head, samples)
    for row in before:
        bad = set.union(*(failures(row[d]) for d in DIRECTIONS))
        if not bad or not bad <= set(GROUPS["alignment"]):
            raise ValueError("CPU initial screen does not confirm alignment-only failure")
    del model
    spec = {
        "kind": KIND,
        "steps_per_head": STEPS,
        "lr": LR,
        "weight_decay": 1e-5,
        "gradient_clip": 5.0,
        "seed": 0,
        "source_sha256": sources,
        "initial_checkpoint_sha256": hashes["residual_head"]["final.pth"],
        "panel": panel,
        "samples": [s["sample"] for s in samples],
        "loss": "unchanged_v7_geometry",
        "base_predictor_updates": 0,
        "parameterization": "four_independent_heads_versus_one_shared_head",
        "best_iterate_selection": False,
        "generator_training_eligible": "hold_not_qualified",
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = args.out_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        if file_sha256(target) != digest:
            raise ValueError("source drift before fitting")
    with (args.out_dir / "spec.json").open("x") as handle:
        json.dump(spec, handle, indent=2, allow_nan=False)
        handle.write("\n")
    started = time.monotonic()
    fits = [(f"independent_{i:02d}", [s]) for i, s in enumerate(samples)] + [("shared", samples)]
    outcomes = {}
    for name, fit_samples in fits:
        head = ResidualVelocityHead()
        head.load_state_dict(initial, strict=True)
        if evaluate(head, samples) != before:
            raise ValueError("different head initialization")
        trace = fit_head(
            head,
            fit_samples,
            steps=STEPS,
            lr=LR,
            callback=lambda i, m, name=name: print(
                f"{name} {i}/{STEPS}: geometry loss {m['total']:.6f}", flush=True
            ),
        )
        after = evaluate(head, samples)
        path = args.out_dir / f"{name}_diagnostic_head.pth"
        torch.save(
            {
                "kind": KIND,
                "spec_sha256": canonical_hash(spec),
                "head_state": head.state_dict(),
                "fit_sequence_ids": [s["sequence_id"] for s in fit_samples],
                "recorded_updates": len(trace),
                "generator_training_eligible": "hold_not_qualified",
            },
            path,
        )
        restored = ResidualVelocityHead()
        restored.load_state_dict(torch.load(path, weights_only=True)["head_state"], strict=True)
        if evaluate(restored, samples) != after:
            raise ValueError("checkpoint reload metrics differ")
        outcomes[name] = {
            "fit_sequence_ids": [s["sequence_id"] for s in fit_samples],
            "trace": trace,
            "after_all_four": after,
            "checkpoint_sha256": file_sha256(path),
            "reload_metrics_equal": True,
            "changed_tensors": sum(
                not torch.equal(initial[k], v) for k, v in head.state_dict().items()
            ),
        }
    for s in samples:
        _, digest = read_batch(args.cache_root, s["sample"]["selection"])
        if digest != s["sample"]["array_sha256"]:
            raise ValueError("used cache changed")
    if pilot_hashes(args.run_root) != hashes:
        raise ValueError("completed pilot artifact drift")
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("diagnostic source drift")
    summary = []
    for i, sample in enumerate(samples):
        single = outcomes[f"independent_{i:02d}"]["after_all_four"][i]
        shared = outcomes["shared"]["after_all_four"][i]
        summary.append(
            {
                "sequence_id": sample["sequence_id"],
                "initial_pass": joint(before[i]),
                "independent_pass": joint(single),
                "shared_pass": joint(shared),
                "independent_failures": sorted(
                    set.union(*(failures(single[d]) for d in DIRECTIONS))
                ),
                "shared_failures": sorted(set.union(*(failures(shared[d]) for d in DIRECTIONS))),
            }
        )
    report = {
        "kind": KIND,
        "device": "cpu",
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "source_sha256": sources,
        "spec_sha256": canonical_hash(spec),
        "pilot_artifacts_sha256": hashes,
        "before": before,
        "outcomes": outcomes,
        "summary": summary,
        "elapsed_seconds": time.monotonic() - started,
        "torch_version": str(torch.__version__),
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "Same-frame supervised fitting, not generalization or physical accuracy.",
            "Four independent heads have more total parameters than one shared head.",
            "Each fit observation is presented200times; no best-iterate selection.",
            "Independent head selection requires sequence identity; not deployable.",
            "Filename-prefix diversity is not verified subject/recording independence.",
            "No global train/validation qualification or generator approval.",
        ],
    }
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
