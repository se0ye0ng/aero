"""Train-annotation-assisted SVF refinement feasibility, NOT model performance.

Fit a small residual control grid independently to each train observation. The
image-conditioned predictor is frozen. Final fits are measured with unchanged
qualification predicates, but fitting and scoring the same annotated frame is
an oracle diagnostic, never registration qualification or inference capability.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from aero_ir.registration.geometry import from_superfusion
from aero_ir.registration.protocol_v7 import INTEGRATION_STEPS, training_loss
from aero_ir.registration.qualification_v4 import direction_pass, direction_statistics
from aero_ir.registration.shared_velocity import control_velocity, shared_fields
from aero_ir.utils.manifest import file_sha256
from scripts.train_antiuav300_registration_v7 import load_trained, read_batch, tensors
from scripts.verify_registration_v7 import verify_arm

DIRECTIONS = ("ir_to_rgb_points", "rgb_to_ir_points")


def joint(row):
    return all(direction_pass(row[k]) for k in DIRECTIONS)


def select_observations(rows, per_group):
    """Deterministic first IDs from both saved-pass strata, not best improvements."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    groups = {"failed": [], "passed": []}
    ids = [r["sequence_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicated sequence identity")
    for row in sorted(rows, key=lambda r: r["sequence_id"]):
        group = "passed" if joint(row) else "failed"
        if len(groups[group]) < per_group:
            groups[group].append(row["sequence_id"])
    return groups


def measurements(fields, vb, ib):
    forward, reverse = fields
    with torch.no_grad():
        return dict(zip(DIRECTIONS, (
            direction_statistics(forward, reverse, ib, vb)[0],
            direction_statistics(reverse, forward, vb, ib)[0],
        ), strict=True))


def refine(velocity, visible, infrared, vb, ib, *, steps, grid, lr, callback=None):
    if steps < 1 or grid < 2 or not 0 < lr < float("inf"):
        raise ValueError("invalid refinement settings")
    base = velocity.detach().clone()
    images = (visible.detach(), infrared.detach(), vb.detach(), ib.detach())
    controls = torch.zeros(base.shape[0], 2, grid, grid, dtype=base.dtype,
                           device=base.device, requires_grad=True)
    optimizer = torch.optim.Adam([controls], lr=lr)
    trace = []

    def fields():
        # Fixed bound24 network pixels, taper at image edge, no ROI/GT mask in
        # the transform parameterization. Annotations enter ONLY fitting loss.
        delta = control_velocity(controls, base.shape[2:], max_pixels=24.0)
        return shared_fields(base + delta, INTEGRATION_STEPS)

    with torch.no_grad():
        before = measurements(fields(), vb, ib)
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = training_loss(*images, fields())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite refinement loss")
        loss.backward()
        if controls.grad is None or not torch.isfinite(controls.grad).all():
            raise ValueError("nonfinite/missing control gradient")
        optimizer.step()
        if not torch.isfinite(controls).all():
            raise ValueError("nonfinite controls")
        trace.append({"update": index + 1, "before_update_loss": metrics})
        if callback is not None and (index == 0 or (index + 1) % 25 == 0):
            callback(index + 1, metrics["total"])
    with torch.no_grad():
        final_fields = fields()
        after = measurements(final_fields, vb, ib)
        _, final_loss = training_loss(*images, final_fields)
    return {"before": before, "after": after, "final_loss": final_loss, "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v7_pilot_e10_seed0"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    parser.add_argument("--arm", choices=("geometry", "geometry_mind"), default="geometry")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite diagnostic")
    if args.steps < 1 or args.grid < 2 or args.per_group < 1 or not 0 < args.lr < float("inf"):
        parser.error("invalid budget")
    root = args.run_root / args.arm
    proof = verify_arm(root, args.arm, args.cache_root)
    if proof["status"] != "verified_saved_artifacts_not_replayed":
        parser.error("completed verified pilot required")
    project = Path(__file__).resolve().parents[1]
    sources = {**proof["spec"]["source_sha256"],
               "scripts/probe_registration_v7_refinement.py": file_sha256(__file__),
               "scripts/verify_registration_v7.py": file_sha256(
                   Path(__file__).with_name("verify_registration_v7.py"))}
    screen = json.loads((root / "final_train_screen.json").read_text())
    groups = select_observations(screen["rows"], args.per_group)
    samples = {row["sequence_id"]: sample for row, sample in
               zip(screen["rows"], screen["samples"], strict=True)}
    torch.set_num_threads(1)
    model, _ = load_trained(root / "shared_velocity_e10.pth", torch.device("cpu"))
    model.requires_grad_(False)
    started = time.monotonic()
    results = []
    for group, ids in groups.items():
        for sid in ids:
            sample = samples[sid]
            arrays, digest = read_batch(args.cache_root, sample["selection"])
            if digest != sample["array_sha256"]:
                raise ValueError("selected train input changed")
            vis, ir, vb, ib = tensors(arrays, torch.device("cpu"))
            with torch.no_grad():
                velocity = from_superfusion(model.predictor(
                    ir, vis, direction="visible_to_infrared"))
            result = refine(velocity, vis, ir, vb, ib, steps=args.steps,
                            grid=args.grid, lr=args.lr,
                            callback=lambda i, loss, sid=sid: print(
                                f"{sid} update {i}/{args.steps} loss={loss:.6f}", flush=True))
            results.append({"sequence_id": sid, "saved_stratum": group,
                            "sample": sample, **result})
            print(f"{sid} joint {joint(result['before'])} -> {joint(result['after'])}",
                  flush=True)
    for name, digest in proof["artifacts_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("pilot artifact changed during probe")
    if any(file_sha256(project / name) != digest for name, digest in sources.items()):
        raise ValueError("source changed during probe")
    report = {
        "kind": "annotation_assisted_per_frame_refinement_not_model_performance",
        "generator_training_eligible": "hold_not_qualified",
        "fit_split": "train", "validation_or_test_access": "none",
        "annotations_enter_optimization": True,
        "predictor_parameter_updates": 0,
        "settings": {"steps_per_frame": args.steps, "control_grid": args.grid,
                     "learning_rate": args.lr, "max_residual_velocity_pixels": 24,
                     "loss": "unchanged_v7_geometry_without_MIND",
                     "selection": "first_lexicographic_IDs_per_saved_pass_stratum",
                     "best_iterate_selection": False},
        "selected_groups": groups,
        "arm": args.arm,
        "source_sha256": sources,
        "pilot_artifacts_sha256": proof["artifacts_sha256"],
        "rows": results,
        "summary": {g: {"frames": sum(r["saved_stratum"] == g for r in results),
                         "before_pass": sum(joint(r["before"]) for r in results
                                            if r["saved_stratum"] == g),
                         "after_pass": sum(joint(r["after"]) for r in results
                                           if r["saved_stratum"] == g)} for g in groups},
        "runtime": {"device": "cpu", "torch": str(torch.__version__),
                    "elapsed_seconds": time.monotonic() - started},
        "limitations": ["Same-frame box fitting and scoring, not independent accuracy.",
                        "Small stratified panel, not a dataset-wide pass-rate estimate.",
                        "Positive outcome does not show image-conditioned learnability.",
                        "Negative outcome does not prove representational impossibility.",
                        "No refined transforms exported for generator use."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
