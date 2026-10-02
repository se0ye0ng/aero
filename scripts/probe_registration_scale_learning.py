"""Matched CPU feasibility fit: geometry versus geometry+scale consistency.

Four selected train frames fit the head; four others only probe this fit. All
eight were previously used in the main training split, so this is not held-out
generalization, physical qualification, or a replacement for a full experiment.
"""

import argparse
import copy
import json
import math
import shutil
import statistics
import time
from pathlib import Path

import torch

from aero_ir.registration.geometry import centre_grid, from_superfusion, sampling_map, valid_support
from aero_ir.registration.protocol_v7 import training_loss
from aero_ir.registration.residual_velocity import ResidualVelocityHead, image_context
from aero_ir.registration.scale_equivariance import source_scale_loss
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.probe_registration_scale_response import SCALES, affine, scale_response, scaled_image
from scripts.probe_registration_v7_refinement import joint, measurements
from scripts.render_registration_v7 import selected_cases
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import ARMS, load_checkpoint

TRAIN_SCALES = SCALES[1:]
FIT_ARMS = {"geometry_only": 0.0, "geometry_plus_scale": 0.1}
STEPS = 24
LR = 1e-4


def cached_context(model, visible, infrared):
    with torch.no_grad():
        velocity = from_superfusion(
            model.base_model.predictor(infrared, visible, direction="visible_to_infrared")
        )
        return {"context": image_context(visible, infrared, velocity), "velocity": velocity}


def role_selection(groups):
    """Freeze one fit and one probe ID per previously defined final-pass stratum."""
    if len(groups) != 4 or any(len(ids) < 2 for ids in groups.values()):
        raise ValueError("two observations in each of four strata required")
    selected = [
        (group, role, sid)
        for group, ids in groups.items()
        for role, sid in zip(("fit", "probe"), sorted(ids)[:2], strict=True)
    ]
    if len({sid for _, _, sid in selected}) != 8:
        raise ValueError("duplicate fit/probe sequence identity")
    return selected


def fixed_response(base, prediction, initial, box, centre, sx, sy):
    """Compare equivariance on common INITIAL-reference support across both arms."""
    result = scale_response(base, prediction, box, centre, sx, sy)
    h, w = initial.shape[-2:]
    original = sampling_map(initial)
    mask = valid_support(original, h, w) & valid_support(affine(original, centre, sx, sy), h, w)
    grid = (centre_grid(initial) + 1) / 2
    mask &= ((grid - box[:, None, None, :2]).abs() <= box[:, None, None, 2:] / 2).all(-1)
    expected = affine(sampling_map(base), centre, sx, sy)
    error = torch.linalg.vector_norm(
        (sampling_map(prediction) - expected) * initial.new_tensor([w / 2, h / 2]), dim=-1
    )
    count = int(mask.sum())
    finite = bool(torch.isfinite(base).all() and torch.isfinite(prediction).all())
    result["fixed_initial_roi"] = {
        "observed_pixels": count,
        "median_error_pixels": float(error[mask].median()) if count and finite else None,
        "p95_error_pixels": float(error[mask].quantile(0.95)) if count and finite else None,
        "current_base_valid_fraction": float(
            valid_support(sampling_map(base), h, w)[mask].float().mean()
        )
        if count
        else None,
        "current_prediction_valid_fraction": float(
            valid_support(sampling_map(prediction), h, w)[mask].float().mean()
        )
        if count
        else None,
    }
    return result


def evaluate(head, samples):
    rows = []
    with torch.no_grad():
        for sample in samples:
            fields = head.fields(sample["context"], sample["velocity"])
            _, geometric = training_loss(
                sample["visible"], sample["infrared"], sample["vb"], sample["ib"], fields
            )
            responses = []
            for scale, variant in zip(TRAIN_SCALES, sample["variants"], strict=True):
                prediction, _ = head.fields(variant["context"], variant["velocity"])
                responses.append(
                    {
                        "scale_xy": scale,
                        "response": fixed_response(
                            fields[0],
                            prediction,
                            sample["initial_forward"],
                            sample["ib"],
                            sample["centre"],
                            *scale,
                        ),
                    }
                )
            rows.append(
                {
                    "sequence_id": sample["sequence_id"],
                    "role": sample["role"],
                    "stratum": sample["stratum"],
                    "geometry_loss": geometric,
                    **measurements(fields, sample["vb"], sample["ib"]),
                    "scale_responses": responses,
                }
            )
    return rows


def fit(head, samples, *, weight, steps=STEPS, callback=None):
    if not samples or any(s["role"] != "fit" for s in samples):
        raise ValueError("only fit-role observations may enter optimizer")
    if not math.isfinite(weight) or weight < 0 or type(steps) is not int or steps < 1:
        raise ValueError("invalid weight or update budget")
    if len({s["sequence_id"] for s in samples}) != len(samples):
        raise ValueError("duplicate fit sequence")
    keys = ("visible", "infrared", "vb", "ib", "context", "velocity")
    data = {k: torch.cat([s[k] for s in samples]).detach() for k in keys}
    variants = [
        {
            k: torch.cat([s["variants"][i][k] for s in samples]).detach()
            for k in ("context", "velocity")
        }
        for i in range(len(TRAIN_SCALES))
    ]
    centres = torch.stack([s["centre"] for s in samples]).detach()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-5)
    head.train()
    trace = []
    for index in range(steps):
        variant = variants[index % len(variants)]
        sx, sy = TRAIN_SCALES[index % len(variants)]
        optimizer.zero_grad(set_to_none=True)
        fields = head.fields(data["context"], data["velocity"])
        geometric, geometry_metrics = training_loss(
            data["visible"], data["infrared"], data["vb"], data["ib"], fields
        )
        # Both arms execute the same augmented forward; only its loss weight differs.
        prediction, _ = head.fields(variant["context"], variant["velocity"])
        scale_loss, support = source_scale_loss(
            fields[0],
            prediction,
            data["ib"],
            centres,
            centres.new_tensor([[sx, sy]]).expand(len(samples), -1),
        )
        total = geometric + weight * scale_loss
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite combined objective")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        trace.append(
            {
                "update": index + 1,
                "scale_xy": [sx, sy],
                "fit_sequence_ids": [s["sequence_id"] for s in samples],
                "before_update_geometry": geometry_metrics,
                "before_update_scale_loss": float(scale_loss.detach()),
                "before_update_total": float(total.detach()),
                "support": support,
                "gradient_norm": float(norm),
            }
        )
        if callback and ((index + 1) % 6 == 0 or index == 0):
            callback(trace[-1])
    if any(not torch.isfinite(p).all() for p in head.parameters()):
        raise FloatingPointError("nonfinite final head")
    return trace, optimizer.state_dict()


def summarize(rows):
    result = {}
    for role in ("fit", "probe"):
        selected = [r for r in rows if r["role"] == role]
        responses = [
            s["response"]["fixed_initial_roi"] for r in selected for s in r["scale_responses"]
        ]
        values = [
            r["median_error_pixels"] for r in responses if r["median_error_pixels"] is not None
        ]
        result[role] = {
            "observations": len(selected),
            "joint_passes": sum(joint(r) for r in selected),
            "mean_geometry_loss": statistics.mean(r["geometry_loss"]["total"] for r in selected),
            "scale_observations": len(responses),
            "measured_scale_rois": len(values),
            "median_of_scale_roi_median_error_pixels": statistics.median(values)
            if values
            else None,
        }
    return result


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
        p.error("fresh output directory required; completed pilots will not be overwritten")
    hashes = pilot_hashes(args.run_root)
    compare(args.run_root, args.cache_root)
    sources = audit_sources()
    for name in (
        "scripts/probe_registration_scale_learning.py",
        "scripts/probe_registration_scale_response.py",
        "scripts/probe_registration_v7_refinement.py",
        "scripts/render_registration_v7.py",
    ):
        sources[name] = file_sha256(name)
    screens = [
        json.loads((args.run_root / arm / "final_train_screen.json").read_text()) for arm in ARMS
    ]
    groups, _ = selected_cases(*(s["rows"] for s in screens))
    selected = role_selection(groups)
    stored = {
        r["sequence_id"]: s for r, s in zip(screens[0]["rows"], screens[0]["samples"], strict=True)
    }
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device("cpu")
    model, _ = load_checkpoint(args.run_root / "residual_head" / "final.pth", device)
    model.requires_grad_(False)
    initial = copy.deepcopy(model.head.state_dict())
    samples = []
    started = time.monotonic()
    with torch.no_grad():
        for group, role, sid in selected:
            arrays, digest = read_batch(args.cache_root, stored[sid]["selection"])
            if digest != stored[sid]["array_sha256"]:
                raise ValueError("selected cache observation differs")
            visible, infrared, vb, ib = tensors(arrays, device)
            centre = 2 * vb[0, :2] - 1
            original = cached_context(model, visible, infrared)
            # The frozen predictor/contexts are reused, not optimized per image.
            variants = [
                cached_context(model, scaled_image(visible, centre, *scale), infrared)
                for scale in TRAIN_SCALES
            ]
            samples.append(
                {
                    "sequence_id": sid,
                    "role": role,
                    "stratum": group,
                    "sample": stored[sid],
                    "visible": visible,
                    "infrared": infrared,
                    "vb": vb,
                    "ib": ib,
                    "centre": centre,
                    **original,
                    "initial_forward": model.head.fields(original["context"], original["velocity"])[
                        0
                    ],
                    "variants": variants,
                }
            )
            print(f"cached frozen image contexts: {sid} ({role})", flush=True)
    before = evaluate(model.head, samples)
    del model
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = args.out_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        if file_sha256(target) != digest:
            raise ValueError("source changed before snapshot")
    outcomes = {}
    for arm, weight in FIT_ARMS.items():
        head = ResidualVelocityHead()
        head.load_state_dict(initial, strict=True)
        if evaluate(head, samples) != before:
            raise ValueError("unmatched initialization")
        trace, optimizer = fit(
            head,
            [s for s in samples if s["role"] == "fit"],
            weight=weight,
            callback=lambda r, arm=arm: print(
                f"{arm} update {r['update']}/{STEPS} loss={r['before_update_total']:.6f}",
                flush=True,
            ),
        )
        after = evaluate(head, samples)
        path = args.out_dir / f"{arm}_diagnostic_head.pth"
        torch.save(
            {
                "kind": "scale_loss_tiny_cpu_diagnostic_not_completed_pilot",
                "head_state": head.state_dict(),
                "optimizer": optimizer,
                "updates": len(trace),
                "weight": weight,
                "base_pilot_checkpoint_sha256": hashes["residual_head"]["final.pth"],
                "generator_training_eligible": "hold_not_qualified",
            },
            path,
        )
        restored = ResidualVelocityHead()
        restored.load_state_dict(torch.load(path, weights_only=True)["head_state"], strict=True)
        if evaluate(restored, samples) != after:
            raise ValueError("checkpoint reload differs")
        outcomes[arm] = {
            "weight": weight,
            "trace": trace,
            "after": after,
            "summary": summarize(after),
            "checkpoint_sha256": file_sha256(path),
            "reload_metrics_equal": True,
            "changed_head_tensors": sum(
                not torch.equal(initial[k], v) for k, v in head.state_dict().items()
            ),
            "optimizer_steps": sorted({int(v["step"]) for v in optimizer["state"].values()}),
            "data_and_scale_schedule_signature": canonical_hash(
                [{k: r[k] for k in ("update", "scale_xy", "fit_sequence_ids")} for r in trace]
            ),
        }
    if len({v["data_and_scale_schedule_signature"] for v in outcomes.values()}) != 1:
        raise ValueError("different arm schedules")
    if pilot_hashes(args.run_root) != hashes:
        raise ValueError("completed pilot artifact drift")
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("diagnostic source drift")
    report = {
        "kind": "matched_scale_consistency_cpu_learning_diagnostic",
        "generator_training_eligible": "hold_not_qualified",
        "source_sha256": sources,
        "pilot_artifacts_sha256": hashes,
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "device": "cpu",
        "settings": {
            "updates_per_arm": STEPS,
            "lr": LR,
            "batch_size": 4,
            "weights": FIT_ARMS,
            "weight_decay": 1e-5,
            "gradient_clip": 5.0,
            "scales": TRAIN_SCALES,
            "best_iterate_selection": False,
            "teacher": "current_original_forward_map_stop_gradient",
            "selection": "first_sorted_fit_second_probe_per_paired_final_stratum",
        },
        "samples": [
            {k: s[k] for k in ("sequence_id", "role", "stratum", "sample")} for s in samples
        ],
        "before": before,
        "before_summary": summarize(before),
        "arms": outcomes,
        "elapsed_seconds": time.monotonic() - started,
        "torch_version": str(torch.__version__),
        "limitations": [
            "Four fit/four probe observations selected from the train split.",
            "Probe excluded only from these updates, not historical training.",
            "Self-relative consistency is not correct physical correspondence.",
            "Original joint predicates unchanged; synthetic response is insufficient.",
            "Not a full pilot, seed replication or final300-epoch experiment.",
        ],
    }
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "before": report["before_summary"],
                **{arm: o["summary"] for arm, o in outcomes.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
