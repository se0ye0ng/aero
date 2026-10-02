"""One balanced CPU training sweep: unchanged geometry versus scale regularization.

All train160 midpoint observations fit both arms once. Quarter-position frames
probe this sweep only; their sequences were used by the original training. No
validation/test data, physical GT, checkpoint selection or qualification approval.
"""

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import torch

from aero_ir.registration.geometry import centre_grid, sampling_map, valid_support
from aero_ir.registration.protocol_v7 import training_loss
from aero_ir.registration.residual_velocity import ResidualVelocityHead
from aero_ir.registration.scale_equivariance import source_scale_loss
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.probe_registration_scale_learning import FIT_ARMS, LR, TRAIN_SCALES, cached_context
from scripts.probe_registration_scale_response import scaled_image
from scripts.probe_registration_v7_refinement import measurements
from scripts.screen_registration_scale_learning import paired_summary
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import load_checkpoint

BATCH_SIZE = 4
KIND = "balanced_train160_single_sweep_scale_diagnostic"


def panels(screen, shards):
    shard_counts = {s["sequence_id"]: s["pairs"] for s in shards}
    rows = screen["rows"]
    if len(shard_counts) != len(shards) or len({r["sequence_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sequence identity")
    if set(shard_counts) != {r["sequence_id"] for r in rows}:
        raise ValueError("screen must cover every train cache shard")
    fit, probe = [], []
    for row, sample in zip(rows, screen["samples"], strict=True):
        sid = row["sequence_id"]
        count = shard_counts[sid]
        if count < 4 or sample["selection"] != [[sid, count // 2]]:
            raise ValueError("expected a valid midpoint and distinct quarter-position probe")
        fit.append({"sequence_id": sid, "role": "fit", **sample})
        probe.append({"sequence_id": sid, "role": "probe", "selection": [[sid, count // 4]]})
    return sorted(fit, key=lambda s: s["sequence_id"]), sorted(
        probe, key=lambda s: s["sequence_id"]
    )


def schedule(samples, *, seed=0):
    if (
        not samples
        or len(samples) % BATCH_SIZE
        or any(s["role"] != "fit" for s in samples)
        or len({s["sequence_id"] for s in samples}) != len(samples)
    ):
        raise ValueError("unique fit-only samples divisible by batch size required")
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    return [
        {
            "samples": ordered[i : i + BATCH_SIZE],
            "scale_xy": TRAIN_SCALES[(i // BATCH_SIZE) % len(TRAIN_SCALES)],
        }
        for i in range(0, len(ordered), BATCH_SIZE)
    ]


def measured_scale_term(reference, prediction, boxes, centres, scales):
    """Keep ALL geometry observations; explicitly log missing auxiliary evidence.

    The measured auxiliary losses sum over eligible samples and divide by the full
    batch size. Missing evidence contributes no added constraint, not a zero error
    or a passing measurement. Underlying source_scale_loss still rejects empty ROIs.
    """
    if not all(torch.isfinite(t).all() for t in (reference, prediction, boxes, centres, scales)):
        raise ValueError("nonfinite balanced scale input")
    if (scales <= 0).any() or (boxes[:, 2:] <= 0).any():
        raise ValueError("positive scales and extents required")
    with torch.no_grad():
        mapped = sampling_map(reference.detach())
        centre = centres[:, None, None]
        expected = centre + (mapped - centre) * scales[:, None, None]
        h, w = reference.shape[-2:]
        grid = (centre_grid(reference) + 1) / 2
        roi = ((grid - boxes[:, None, None, :2]).abs() <= boxes[:, None, None, 2:] / 2).all(-1)
        counts = (roi & valid_support(mapped, h, w) & valid_support(expected, h, w)).sum((1, 2))
        eligible = counts > 0
    if eligible.any():
        loss, _ = source_scale_loss(
            reference[eligible],
            prediction[eligible],
            boxes[eligible],
            centres[eligible],
            scales[eligible],
        )
        effective = loss * eligible.float().mean()
        measured = float(loss.detach())
    else:
        effective = prediction.sum() * 0
        measured = None
    return effective, {
        "eligible_samples": eligible.tolist(),
        "target_pixels": counts.tolist(),
        "eligible_mean_loss": measured,
        "effective_full_batch_term": float(effective.detach()),
    }


def train_pair(heads, batches, model, cache):
    optimizers = {
        arm: torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-5)
        for arm, head in heads.items()
    }
    traces = {arm: [] for arm in heads}
    for index, batch in enumerate(batches, 1):
        inputs, context_parts, augmented_parts = [], [], []
        for sample in batch["samples"]:
            arrays, digest = read_batch(cache, sample["selection"])
            if digest != sample["array_sha256"]:
                raise ValueError("fit observation changed")
            visible, infrared, vb, ib = tensors(arrays, torch.device("cpu"))
            inputs.append((visible, infrared, vb, ib))
            context_parts.append(cached_context(model, visible, infrared))
            augmented = scaled_image(visible, 2 * vb[0, :2] - 1, *batch["scale_xy"])
            augmented_parts.append(cached_context(model, augmented, infrared))
        visible, infrared, vb, ib = (torch.cat(values) for values in zip(*inputs, strict=True))
        contexts = {k: torch.cat([s[k] for s in context_parts]) for k in ("context", "velocity")}
        augmented = {k: torch.cat([s[k] for s in augmented_parts]) for k in ("context", "velocity")}
        for arm, head in heads.items():
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            fields = head.fields(contexts["context"], contexts["velocity"])
            geometric, metrics = training_loss(visible, infrared, vb, ib, fields)
            prediction, _ = head.fields(augmented["context"], augmented["velocity"])
            extra, support = measured_scale_term(
                fields[0],
                prediction,
                ib,
                2 * vb[:, :2] - 1,
                vb.new_tensor([batch["scale_xy"]]).expand(len(inputs), -1),
            )
            loss = geometric + FIT_ARMS[arm] * extra
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite balanced objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            traces[arm].append(
                {
                    "update": index,
                    "samples": batch["samples"],
                    "scale_xy": batch["scale_xy"],
                    "geometry": metrics,
                    "scale": support,
                    "before_update_total": float(loss.detach()),
                    "gradient_norm": float(norm),
                }
            )
        if index == 1 or index % 5 == 0:
            print(f"both CPU arms updated: {index}/{len(batches)}", flush=True)
    return traces, {a: o.state_dict() for a, o in optimizers.items()}


def screen_panel(heads, samples, model, cache):
    rows = {arm: [] for arm in heads}
    with torch.no_grad():
        for index, sample in enumerate(samples, 1):
            arrays, digest = read_batch(cache, sample["selection"])
            if sample.get("array_sha256", digest) != digest:
                raise ValueError("screen observation changed")
            sample["array_sha256"] = digest
            visible, infrared, vb, ib = tensors(arrays, torch.device("cpu"))
            contexts = cached_context(model, visible, infrared)
            for arm, head in heads.items():
                fields = head.fields(contexts["context"], contexts["velocity"])
                rows[arm].append(
                    {
                        "sequence_id": sample["sequence_id"],
                        "sample": sample,
                        **measurements(fields, vb, ib),
                    }
                )
            if index == 1 or index % 40 == 0:
                print(f"CPU {sample['role']} screen: {index}/{len(samples)}", flush=True)
    return rows


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
        "scripts/probe_registration_balanced_scale_learning.py",
        "scripts/probe_registration_scale_learning.py",
        "scripts/probe_registration_scale_response.py",
        "scripts/probe_registration_v7_refinement.py",
        "scripts/render_registration_v7.py",
        "scripts/screen_registration_scale_learning.py",
    ):
        sources[name] = file_sha256(name)
    screen = json.loads((args.run_root / "residual_head" / "final_train_screen.json").read_text())
    manifest_hash = file_sha256(args.cache_root / "manifest.json")
    shards = json.loads((args.cache_root / "manifest.json").read_text())["shards"]
    fit_samples, probe_samples = panels(screen, shards)
    if len(fit_samples) != 160:
        p.error("frozen160-sequence train coverage required")
    batches = schedule(fit_samples)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, _ = load_checkpoint(args.run_root / "residual_head" / "final.pth", torch.device("cpu"))
    model.requires_grad_(False)
    initial = copy.deepcopy(model.head.state_dict())
    heads = {}
    for arm in FIT_ARMS:
        head = ResidualVelocityHead()
        head.load_state_dict(initial, strict=True)
        heads[arm] = head.train()
    spec = {
        "kind": KIND,
        "batch_size": BATCH_SIZE,
        "updates_per_arm": len(batches),
        "lr": LR,
        "weight_decay": 1e-5,
        "gradient_clip": 5.0,
        "seed": 0,
        "weights": FIT_ARMS,
        "source_sha256": sources,
        "schedule": batches,
        "schedule_sha256": canonical_hash(batches),
        "cache_manifest_sha256": manifest_hash,
        "initial_checkpoint_sha256": hashes["residual_head"]["final.pth"],
        "probe_selection": "floor(shard_pairs/4); excluded only from this sweep",
        "best_iterate_selection": False,
        "generator_training_eligible": "hold_not_qualified",
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = args.out_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        if file_sha256(target) != digest:
            raise ValueError("source drift before fit")
    with (args.out_dir / "spec.json").open("x") as handle:
        json.dump(spec, handle, indent=2, allow_nan=False)
        handle.write("\n")
    started = time.monotonic()
    traces, optimizers = train_pair(heads, batches, model, args.cache_root)
    all_heads = {"initial": model.head, **heads}
    rows = {
        role: screen_panel(all_heads, panel, model, args.cache_root)
        for role, panel in (("fit_midpoints", fit_samples), ("probe_quarters", probe_samples))
    }
    summaries = {
        role: {
            arm: paired_summary(values["initial"], candidate) for arm, candidate in values.items()
        }
        for role, values in rows.items()
    }
    checkpoints = {}
    for arm, head in heads.items():
        steps = {int(v["step"]) for v in optimizers[arm]["state"].values()}
        if steps != {len(batches)} or any(not torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("invalid final head/budget")
        path = args.out_dir / f"{arm}_diagnostic_head.pth"
        torch.save(
            {
                "kind": KIND,
                "spec_sha256": canonical_hash(spec),
                "head_state": head.state_dict(),
                "optimizer": optimizers[arm],
                "updates": len(batches),
                "arm": arm,
                "generator_training_eligible": "hold_not_qualified",
            },
            path,
        )
        restored = ResidualVelocityHead()
        restored.load_state_dict(torch.load(path, weights_only=True)["head_state"], strict=True)
        if any(not torch.equal(v, restored.state_dict()[k]) for k, v in head.state_dict().items()):
            raise ValueError("checkpoint tensor roundtrip differs")
        checkpoints[arm] = {
            "sha256": file_sha256(path),
            "tensor_roundtrip_equal": True,
            "optimizer_steps": sorted(steps),
        }
    for sample in fit_samples + probe_samples:
        _, digest = read_batch(args.cache_root, sample["selection"])
        if digest != sample["array_sha256"]:
            raise ValueError("used cache bytes changed")
    if any(not torch.equal(v, model.head.state_dict()[k]) for k, v in initial.items()):
        raise ValueError("initial head changed")
    if (
        pilot_hashes(args.run_root) != hashes
        or file_sha256(args.cache_root / "manifest.json") != manifest_hash
    ):
        raise ValueError("pilot/cache drift")
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("source drift")
    report = {
        "kind": KIND,
        "spec_sha256": canonical_hash(spec),
        "source_sha256": sources,
        "pilot_artifacts_sha256": hashes,
        "summary": summaries,
        "rows": rows,
        "cpu_initial_vs_saved_gpu": paired_summary(
            screen["rows"], rows["fit_midpoints"]["initial"]
        ),
        "traces": traces,
        "checkpoints": checkpoints,
        "elapsed_seconds": time.monotonic() - started,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "validation_or_test_access": "none",
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "One balanced sweep, not a full GPU pilot or final300-epoch run.",
            "Both midpoint fit and quarter probe are official training sequences.",
            "Quarter probes withheld only from these40updates, not earlier training.",
            "No independent physical correspondence accuracy or qualification.",
            "Auxiliary-ineligible observations retain their full original geometry loss.",
        ],
    }
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                role: {
                    arm: {k: v for k, v in s.items() if not k.endswith("ids")}
                    for arm, s in values.items()
                }
                for role, values in summaries.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
