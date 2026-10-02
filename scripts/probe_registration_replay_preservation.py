"""Matched CPU hard-case/replay fit, with or without frozen-output preservation.

Four fixed train failures are revisited alongside cycling previously passing
train frames. No validation/test labels, per-frame head routing, new physical GT,
early stopping or generator qualification. This is not the full-training pilot.
"""

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import torch

from aero_ir.registration.output_preservation import GLOBAL_WEIGHT, preservation_loss
from aero_ir.registration.protocol_v7 import training_loss
from aero_ir.registration.residual_velocity import ResidualVelocityHead
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.probe_registration_balanced_scale_learning import panels, screen_panel
from scripts.probe_registration_fit_capacity import LR, STEPS, alignment_failure_panel
from scripts.probe_registration_scale_learning import cached_context
from scripts.probe_registration_v7_refinement import joint, measurements
from scripts.screen_registration_scale_learning import paired_summary
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import load_checkpoint

ARMS = {"geometry_replay": 0.0, "geometry_replay_preserve": 1.0}
KIND = "hard_case_replay_output_preservation_cpu_diagnostic"
REPLAY_BATCH = 4


def replay_schedule(passing_ids, *, steps=STEPS, seed=0):
    if (
        len(passing_ids) < REPLAY_BATCH
        or len(set(passing_ids)) != len(passing_ids)
        or type(steps) is not int
        or steps < 1
    ):
        raise ValueError("unique replay identities and positive update budget required")
    ids = sorted(passing_ids)
    random.Random(seed).shuffle(ids)
    return [
        [ids[(i * REPLAY_BATCH + j) % len(ids)] for j in range(REPLAY_BATCH)] for i in range(steps)
    ]


def stack_samples(samples):
    if not samples or any(s["role"] not in ("hard_fit", "passing_replay") for s in samples):
        raise ValueError("probe samples cannot enter optimizer")
    values = {
        key: torch.cat([s[key] for s in samples]).detach()
        for key in ("visible", "infrared", "vb", "ib", "context", "velocity")
    }
    values["teacher"] = tuple(
        torch.cat([s["teacher"][d] for s in samples]).detach() for d in range(2)
    )
    return values


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
        "scripts/probe_registration_replay_preservation.py",
        "scripts/probe_registration_fit_capacity.py",
        "scripts/probe_registration_residual_learning.py",
        "scripts/analyze_registration_constraints.py",
        "scripts/probe_registration_balanced_scale_learning.py",
        "scripts/probe_registration_scale_learning.py",
        "scripts/probe_registration_scale_response.py",
        "scripts/probe_registration_v7_refinement.py",
        "scripts/render_registration_v7.py",
        "scripts/screen_registration_scale_learning.py",
    ):
        sources[name] = file_sha256(name)
    screen = json.loads((args.run_root / "residual_head" / "final_train_screen.json").read_text())
    hard_panel = alignment_failure_panel(screen["rows"])
    hard_ids = [r["sequence_id"] for r in hard_panel]
    passing_ids = sorted(r["sequence_id"] for r in screen["rows"] if joint(r))
    if set(hard_ids) & set(passing_ids):
        raise ValueError("hard cases and passing replay overlap")
    schedule = replay_schedule(passing_ids)
    stored = {r["sequence_id"]: s for r, s in zip(screen["rows"], screen["samples"], strict=True)}
    manifest_hash = file_sha256(args.cache_root / "manifest.json")
    shards = json.loads((args.cache_root / "manifest.json").read_text())["shards"]
    midpoint, quarter = panels(screen, shards)
    if len(midpoint) != 160 or len(passing_ids) != 117:
        p.error("expected original160-sequence pilot with117 midpoint passes")
    for s in midpoint:
        s["role"] = (
            "hard_fit"
            if s["sequence_id"] in hard_ids
            else "passing_replay"
            if s["sequence_id"] in passing_ids
            else "not_in_additional_fit"
        )
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, _ = load_checkpoint(args.run_root / "residual_head" / "final.pth", torch.device("cpu"))
    model.requires_grad_(False)
    initial = copy.deepcopy(model.head.state_dict())
    heads = {}
    for arm in ARMS:
        head = ResidualVelocityHead()
        head.load_state_dict(initial, strict=True)
        heads[arm] = head.train()
    spec = {
        "kind": KIND,
        "updates_per_arm": STEPS,
        "lr": LR,
        "weight_decay": 1e-5,
        "gradient_clip": 5.0,
        "seed": 0,
        "hard_batch": len(hard_ids),
        "replay_batch": REPLAY_BATCH,
        "weights": ARMS,
        "preservation_global_weight": GLOBAL_WEIGHT,
        "geometry_weighting": "half hard mean plus half replay mean",
        "hard_panel": hard_panel,
        "passing_ids": passing_ids,
        "replay_schedule": schedule,
        "schedule_sha256": canonical_hash({"hard_ids": hard_ids, "replay": schedule}),
        "source_sha256": sources,
        "cache_manifest_sha256": manifest_hash,
        "initial_checkpoint_sha256": hashes["residual_head"]["final.pth"],
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
    # At most121 fixed train frames (~0.85GB of CPU tensors); no full video loading.
    cache = {}

    def observation(sid):
        if sid not in cache:
            arrays, digest = read_batch(args.cache_root, stored[sid]["selection"])
            if digest != stored[sid]["array_sha256"]:
                raise ValueError("training cache observation differs")
            visible, infrared, vb, ib = tensors(arrays, torch.device("cpu"))
            context = cached_context(model, visible, infrared)
            with torch.no_grad():
                teacher = model.head.fields(context["context"], context["velocity"])
            if sid in passing_ids and not joint(measurements(teacher, vb, ib)):
                raise ValueError("replay teacher is not CPU-verified passing")
            cache[sid] = {
                "role": "hard_fit" if sid in hard_ids else "passing_replay",
                "visible": visible,
                "infrared": infrared,
                "vb": vb,
                "ib": ib,
                **context,
                "teacher": teacher,
            }
        return cache[sid]

    started = time.monotonic()
    hard = stack_samples([observation(sid) for sid in hard_ids])
    optimizers = {
        a: torch.optim.AdamW(h.parameters(), lr=LR, weight_decay=1e-5) for a, h in heads.items()
    }
    traces = {a: [] for a in heads}
    with (args.out_dir / "trace.jsonl").open("x") as log:
        for index, ids in enumerate(schedule, 1):
            replay = stack_samples([observation(sid) for sid in ids])
            context = torch.cat((hard["context"], replay["context"]))
            velocity = torch.cat((hard["velocity"], replay["velocity"]))
            for arm, head in heads.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                all_fields = head.fields(context, velocity)
                hard_fields = tuple(f[: len(hard_ids)] for f in all_fields)
                replay_fields = tuple(f[len(hard_ids) :] for f in all_fields)
                hard_loss, hm = training_loss(
                    hard["visible"], hard["infrared"], hard["vb"], hard["ib"], hard_fields
                )
                replay_loss, rm = training_loss(
                    replay["visible"], replay["infrared"], replay["vb"], replay["ib"], replay_fields
                )
                preserve, pm = preservation_loss(
                    replay_fields, replay["teacher"], replay["vb"], replay["ib"]
                )
                total = (hard_loss + replay_loss) / 2 + ARMS[arm] * preserve
                if not torch.isfinite(total):
                    raise FloatingPointError("nonfinite replay objective")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                row = {
                    "arm": arm,
                    "update": index,
                    "hard_ids": hard_ids,
                    "replay_ids": ids,
                    "hard_geometry": hm,
                    "replay_geometry": rm,
                    "preservation": pm,
                    "preservation_loss": float(preserve.detach()),
                    "total": float(total.detach()),
                    "gradient_norm": float(norm),
                }
                traces[arm].append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if index == 1 or index % 25 == 0:
                print(f"both replay arms updated: {index}/{STEPS}", flush=True)
    cache.clear()
    checkpoints = {}
    for arm, head in list(heads.items()):
        state = optimizers[arm].state_dict()
        steps = sorted({int(v["step"]) for v in state["state"].values()})
        if steps != [STEPS] or any(not torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("incomplete or nonfinite replay head")
        path = args.out_dir / f"{arm}_diagnostic_head.pth"
        torch.save(
            {
                "kind": KIND,
                "spec_sha256": canonical_hash(spec),
                "arm": arm,
                "head_state": head.state_dict(),
                "optimizer": state,
                "updates": STEPS,
                "generator_training_eligible": "hold_not_qualified",
            },
            path,
        )
        restored = ResidualVelocityHead()
        restored.load_state_dict(torch.load(path, weights_only=True)["head_state"], strict=True)
        if any(not torch.equal(v, restored.state_dict()[k]) for k, v in head.state_dict().items()):
            raise ValueError("head tensor reload differs")
        heads[arm] = restored.eval()
        checkpoints[arm] = {
            "sha256": file_sha256(path),
            "optimizer_steps": steps,
            "tensor_reload_equal": True,
            "screen_uses_reloaded_head": True,
        }
    all_heads = {"initial": model.head, **heads}
    rows = {
        role: screen_panel(all_heads, data, model, args.cache_root)
        for role, data in (("train_midpoints", midpoint), ("train_quarters", quarter))
    }
    summary = {
        role: {a: paired_summary(values["initial"], v) for a, v in values.items()}
        for role, values in rows.items()
    }
    for s in midpoint + quarter:
        _, digest = read_batch(args.cache_root, s["selection"])
        if digest != s["array_sha256"]:
            raise ValueError("used cache bytes changed")
    if any(not torch.equal(v, model.head.state_dict()[k]) for k, v in initial.items()):
        raise ValueError("initial teacher changed")
    if (
        pilot_hashes(args.run_root) != hashes
        or file_sha256(args.cache_root / "manifest.json") != manifest_hash
    ):
        raise ValueError("pilot/cache artifact drift")
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("source drift")
    report = {
        "kind": KIND,
        "device": "cpu",
        "source_sha256": sources,
        "spec_sha256": canonical_hash(spec),
        "pilot_artifacts_sha256": hashes,
        "summary": summary,
        "rows": rows,
        "traces": traces,
        "checkpoints": checkpoints,
        "trace_file_sha256": file_sha256(args.out_dir / "trace.jsonl"),
        "cpu_initial_vs_saved_gpu": paired_summary(
            screen["rows"], rows["train_midpoints"]["initial"]
        ),
        "hard_case_passes": {
            a: sum(joint(r) for r in rs if r["sequence_id"] in hard_ids)
            for a, rs in rows["train_midpoints"].items()
        },
        "elapsed_seconds": time.monotonic() - started,
        "torch_version": str(torch.__version__),
        "validation_or_test_access": "none",
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "All observations are prior training sequences, not held-out accuracy.",
            "Quarter frames excluded only from these additional updates.",
            "Frozen teacher is a weak engineering baseline, not physical GT.",
            "Repeated four-hard-frame fit is not a full-training pilot.",
            "Neither improved box metrics nor output preservation qualifies physical registration.",
        ],
    }
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
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
    print(json.dumps({"hard_case_passes": report["hard_case_passes"]}, indent=2))


if __name__ == "__main__":
    main()
