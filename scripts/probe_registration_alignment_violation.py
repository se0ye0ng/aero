"""Matched CPU loss ablation from completed replay300; never qualification GO.

All160 train midpoints fit both heads. Quarter frames are excluded only from
these additional updates, not from earlier training. YAML defines the experiment.
No GT boxes enter prediction and no original source/checkpoint is modified.
"""

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import torch
import yaml

from aero_ir.registration.alignment_violation import alignment_violation
from aero_ir.registration.protocol_v7 import training_loss
from aero_ir.registration.qualification_v4 import direction_pass, split_report
from aero_ir.registration.residual_velocity import ResidualVelocityHead
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.probe_registration_scale_learning import cached_context
from scripts.probe_registration_v7_refinement import measurements
from scripts.screen_registration_scale_learning import paired_summary
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_replay import compare, load_checkpoint

PROJECT = Path(__file__).resolve().parents[1]
KIND = "registration_alignment_violation_cpu_v1"
HOLD = "hold_not_qualified"
KEYS = ("visible", "infrared", "vb", "ib", "context", "velocity")


def read_config(path):
    cfg = yaml.safe_load(path.read_text())
    fixed = {
        "kind": KIND, "device": "cpu", "initial_arm": "uniform",
        "fit_panel": "all_train_midpoints", "probe_panel": "all_train_quarter_frames",
        "auxiliary_weights": {"geometry_only": 0.0, "geometry_plus_violation": 1.0},
    }
    numeric = {"seed", "updates_per_arm", "batch_size", "learning_rate", "weight_decay",
               "gradient_clip"}
    if (not isinstance(cfg, dict) or set(cfg) != set(fixed) | numeric
            or any(cfg.get(k) != v for k, v in fixed.items())):
        raise ValueError("unsupported diagnostic configuration")
    for k in ("seed", "updates_per_arm", "batch_size"):
        if type(cfg[k]) is not int or cfg[k] < (0 if k == "seed" else 1):
            raise ValueError(f"invalid {k}")
    for k in ("learning_rate", "weight_decay", "gradient_clip"):
        if type(cfg[k]) not in (int, float) or not 0 < cfg[k] < float("inf"):
            raise ValueError(f"invalid {k}")
    if 160 % cfg["batch_size"] or cfg["updates_per_arm"] % (160 // cfg["batch_size"]):
        raise ValueError("complete sweeps of160 midpoints required")
    return cfg


def schedule(ids, updates, batch_size, seed):
    if (not ids or len(set(ids)) != len(ids) or batch_size < 1
            or len(ids) % batch_size or updates < 1):
        raise ValueError("unique full-batch panel required")
    result, epoch = [], 0
    while len(result) < updates:
        ordered = sorted(ids)
        random.Random(seed + epoch).shuffle(ordered)
        result.extend(ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size))
        epoch += 1
    return result[:updates]


def screen(head, samples):
    rows = []
    with torch.no_grad():
        for sample in samples:
            fields = head.fields(sample["context"], sample["velocity"])
            rows.append({"sequence_id": sample["sequence_id"],
                         "frame_index": sample["sample"]["selection"][0][1],
                         **measurements(fields, sample["vb"], sample["ib"])})
    return rows


def pass_map(rows):
    return {r["sequence_id"]: all(direction_pass(r[d]) for d in (
        "ir_to_rgb_points", "rgb_to_ir_points")) for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(
        "configs/experiment/registration_alignment_violation_cpu.yaml"))
    p.add_argument("--run-root", type=Path,
                   default=Path("experiments/registration_replay_e300_seed0"))
    p.add_argument("--base-root", type=Path,
                   default=Path("experiments/registration_residual_pilot_e10_seed0"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        p.error("fresh output directory required; never overwrite a diagnostic")
    cfg = read_config(args.config)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    proof = compare(args.run_root, args.cache_root, args.base_root)
    comparison_path = args.run_root / "comparison.json"
    if json.loads(comparison_path.read_text()) != proof:
        raise ValueError("saved comparison does not reproduce")
    root = args.run_root / cfg["initial_arm"]
    spec = json.loads((root / "run_spec.json").read_text())
    initial_path = root / "final.pth"
    artifact_paths = [comparison_path, initial_path, root / "run_spec.json",
                      args.cache_root / "manifest.json"]
    artifact_paths += [root / f"final_train_div{d}.json" for d in (2, 4)]
    artifacts = {str(path.resolve()): file_sha256(path) for path in artifact_paths}
    sources = dict(spec["source_sha256"])
    for name in (
        "scripts/probe_registration_alignment_violation.py",
        "src/aero_ir/registration/alignment_violation.py",
        "scripts/probe_registration_scale_learning.py",
        "scripts/probe_registration_v7_refinement.py",
        "scripts/probe_registration_scale_response.py",
        "scripts/screen_registration_scale_learning.py",
        str(args.config.resolve().relative_to(PROJECT)),
    ):
        sources[name] = file_sha256(PROJECT / name)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = args.out_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
        if file_sha256(target) != digest:
            raise ValueError("source snapshot changed")
    model, _ = load_checkpoint(initial_path, spec, torch.device("cpu"))
    model.eval().requires_grad_(False)
    panels, before, saved = {}, {}, {}
    for divisor, role in ((2, "fit"), (4, "probe")):
        stored = json.loads((root / f"final_train_div{divisor}.json").read_text())
        saved[role] = stored["rows"]
        panels[role] = []
        for row, sample in zip(stored["rows"], stored["samples"], strict=True):
            arrays, digest = read_batch(args.cache_root, sample["selection"])
            if digest != sample["array_sha256"]:
                raise ValueError("cached observation differs from completed run")
            visible, infrared, vb, ib = tensors(arrays, torch.device("cpu"))
            panels[role].append({"sequence_id": row["sequence_id"], "sample": sample,
                                 "visible": visible, "infrared": infrared, "vb": vb, "ib": ib,
                                 **cached_context(model, visible, infrared)})
        before[role] = screen(model.head, panels[role])
        if pass_map(before[role]) != pass_map(saved[role]):
            raise ValueError("initial CPU pass decisions differ from saved GPU panel")
        count = sum(pass_map(before[role]).values())
        print(f"Prepared and replayed {role}: {count}/160", flush=True)
    by_id = {s["sequence_id"]: s for s in panels["fit"]}
    batches = schedule(list(by_id), cfg["updates_per_arm"], cfg["batch_size"], cfg["seed"])
    experiment = {"config": cfg, "source_sha256": sources, "input_sha256": artifacts,
                  "batch_order": batches, "generator_training_eligible": HOLD}
    (args.out_dir / "spec.json").write_text(json.dumps(experiment, indent=2) + "\n")
    heads = {arm: copy.deepcopy(model.head).requires_grad_(True).train()
             for arm in cfg["auxiliary_weights"]}
    optimizers = {a: torch.optim.AdamW(h.parameters(), lr=cfg["learning_rate"],
                                     weight_decay=cfg["weight_decay"])
                  for a, h in heads.items()}
    trace = []
    with (args.out_dir / "trace.jsonl").open("x") as log:
        for update, ids in enumerate(batches, 1):
            batch = {k: torch.cat([by_id[sid][k] for sid in ids]) for k in KEYS}
            for arm, head in heads.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                fields = head.fields(batch["context"], batch["velocity"])
                geometric, metrics = training_loss(batch["visible"], batch["infrared"],
                                                   batch["vb"], batch["ib"], fields)
                auxiliary = alignment_violation(fields, batch["vb"], batch["ib"])
                loss = geometric + cfg["auxiliary_weights"][arm] * auxiliary
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["gradient_clip"],
                                                     error_if_nonfinite=True)
                optimizer.step()
                row = {"arm": arm, "update": update, "ids": ids, "geometry": metrics,
                       "auxiliary": float(auxiliary.detach()), "total": float(loss.detach()),
                       "gradient_norm": float(norm)}
                trace.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if update == 1 or update % 25 == 0:
                print(f"Both arms: {update}/{len(batches)} updates", flush=True)
    checkpoints = {}
    for arm, head in heads.items():
        steps = {int(v["step"]) for v in optimizers[arm].state.values()}
        if steps != {cfg["updates_per_arm"]} or any(
                not torch.isfinite(v).all() for v in head.state_dict().values()):
            raise ValueError("incomplete/nonfinite fitted head")
        path = args.out_dir / f"{arm}_diagnostic_head.pth"
        torch.save({"kind": KIND, "head_state": head.state_dict(),
                    "optimizer": optimizers[arm].state_dict(),
                    "spec_sha256": canonical_hash(experiment), "generator_training_eligible": HOLD},
                   path)
        restored = ResidualVelocityHead()
        restored.load_state_dict(torch.load(path, weights_only=True)["head_state"], strict=True)
        if any(not torch.equal(v, restored.state_dict()[k]) for k, v in head.state_dict().items()):
            raise ValueError("checkpoint reload mismatch")
        heads[arm] = restored.eval()
        checkpoints[arm] = {"sha256": file_sha256(path), "optimizer_steps": sorted(steps),
                            "tensor_reload_equal": True}
    rows = {role: {"initial": before[role], **{a: screen(h, samples) for a, h in heads.items()}}
            for role, samples in panels.items()}
    summary = {role: {a: {"metrics": split_report(rs),
                         "change_from_initial": paired_summary(before[role], rs)}
                      for a, rs in arms.items()} for role, arms in rows.items()}
    for samples in panels.values():
        for sample in samples:
            _, digest = read_batch(args.cache_root, sample["sample"]["selection"])
            if digest != sample["sample"]["array_sha256"]:
                raise ValueError("cached input changed during diagnostic")
    if any(file_sha256(n) != h for n, h in artifacts.items()) or any(
            file_sha256(PROJECT / n) != h for n, h in sources.items()):
        raise ValueError("source/input drift")
    report = {"kind": KIND, "spec_sha256": canonical_hash(experiment), "config": cfg,
              "source_sha256": sources, "input_sha256": artifacts, "rows": rows,
              "summary": summary, "checkpoints": checkpoints,
              "samples": {r: [{"sequence_id": s["sequence_id"], **s["sample"]}
                               for s in ss] for r, ss in panels.items()},
              "trace_sha256": file_sha256(args.out_dir / "trace.jsonl"),
              "logged_updates_per_arm": len(trace) // len(heads),
              "runtime": {"device": "cpu", "torch": str(torch.__version__),
                          "seconds": time.monotonic() - started},
              "validation_or_test_access": "none", "generator_training_eligible": HOLD,
              "limitations": ["Both panels were exposed to the earlier full training.",
                              "Quarter frames excluded only from these additional updates.",
                              "Box-gate optimization is not independent physical evidence.",
                              "This is a CPU loss diagnostic, not a full training experiment."]}
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({r: {a: v["metrics"]["joint_frame_pass_rate"] for a, v in arms.items()}
                      for r, arms in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
