"""CPU image-conditioned residual learning diagnostic; no qualification claim.

Eight annotated train frames fit one shared head, eight other train frames probe
transfer. The frozen v7 model previously saw these sequences: the probe is NOT an
untouched validation/test set. Boxes enter losses/metrics only, never prediction.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from aero_ir.registration.geometry import from_superfusion
from aero_ir.registration.protocol_v7 import training_loss
from aero_ir.registration.residual_velocity import (
    ARCHITECTURE,
    ResidualVelocityHead,
    image_context,
)
from aero_ir.utils.manifest import file_sha256
from scripts.probe_registration_v7_refinement import joint, measurements, select_observations
from scripts.train_antiuav300_registration_v7 import load_trained, read_batch, tensors
from scripts.verify_registration_v7 import verify_arm


def evaluate(head, samples):
    result = []
    with torch.no_grad():
        for sample in samples:
            fields = head.fields(sample["context"], sample["velocity"])
            result.append({"sequence_id": sample["sequence_id"],
                           "role": sample["role"], "saved_stratum": sample["saved_stratum"],
                           **measurements(fields, sample["vb"], sample["ib"])})
    return result


def fit_head(head, samples, *, steps, lr, callback=None):
    if not samples or steps < 1 or not 0 < lr < float("inf"):
        raise ValueError("invalid fit inputs/budget")
    if any(s["role"] != "fit" for s in samples):
        raise ValueError("probe observation cannot enter optimizer")
    data = {key: torch.cat([s[key] for s in samples]).detach()
            for key in ("visible", "infrared", "vb", "ib", "context", "velocity")}
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    trace = []
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        fields = head.fields(data["context"], data["velocity"])
        loss, metrics = training_loss(data["visible"], data["infrared"], data["vb"],
                                      data["ib"], fields)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite image-conditioned learning loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        trace.append({"update": index+1, "before_update_loss": metrics,
                      "gradient_norm": float(norm)})
        if callback and (index == 0 or (index+1) % 25 == 0):
            callback(index+1, metrics)
    if any(not torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError("nonfinite trained head")
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v7_pilot_e10_seed0"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a fresh output directory")
    if args.steps < 1 or not 0 < args.lr < float("inf"):
        parser.error("invalid budget")
    arm = "geometry"
    root = args.run_root / arm
    proof = verify_arm(root, arm, args.cache_root)
    if proof["status"] != "verified_saved_artifacts_not_replayed":
        parser.error("verified completed v7 control required")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    project = Path(__file__).resolve().parents[1]
    sources = {**proof["spec"]["source_sha256"]}
    for name in ("scripts/probe_registration_residual_learning.py",
                 "scripts/probe_registration_v7_refinement.py",
                 "scripts/verify_registration_v7.py",
                 "src/aero_ir/registration/residual_velocity.py"):
        sources[name] = file_sha256(project / name)
    screen = json.loads((root / "final_train_screen.json").read_text())
    groups = select_observations(screen["rows"], 8)
    if any(len(ids) != 8 for ids in groups.values()):
        parser.error("eight observations in each stratum required")
    stored_samples = {r["sequence_id"]: s for r, s in
                      zip(screen["rows"], screen["samples"], strict=True)}
    model, _ = load_trained(root / "shared_velocity_e10.pth", torch.device("cpu"))
    model.requires_grad_(False)
    started = time.monotonic()
    samples = []
    for group, ids in groups.items():
        for index, sid in enumerate(ids):
            stored = stored_samples[sid]
            arrays, digest = read_batch(args.cache_root, stored["selection"])
            if digest != stored["array_sha256"]:
                raise ValueError("selected train input changed")
            visible, infrared, vb, ib = tensors(arrays, torch.device("cpu"))
            with torch.no_grad():
                velocity = from_superfusion(model.predictor(
                    infrared, visible, direction="visible_to_infrared"))
                context = image_context(visible, infrared, velocity)
            samples.append({"sequence_id": sid, "role": "fit" if index < 4 else "probe",
                            "saved_stratum": group, "sample": stored, "visible": visible,
                            "infrared": infrared, "vb": vb, "ib": ib,
                            "velocity": velocity, "context": context})
    del model
    head = ResidualVelocityHead()
    initial = {name: value.clone() for name, value in head.state_dict().items()}
    before = evaluate(head, samples)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = args.out_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
        if file_sha256(target) != digest:
            raise ValueError("source changed while snapshotting")
    trace = fit_head(head, [s for s in samples if s["role"] == "fit"],
                     steps=args.steps, lr=args.lr,
                     callback=lambda i, m: print(f"update {i}/{args.steps} loss={m['total']:.6f}",
                                                 flush=True))
    after = evaluate(head, samples)
    for name, digest in proof["artifacts_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("v7 artifact changed")
    if any(file_sha256(project / name) != digest for name, digest in sources.items()):
        raise ValueError("source changed")
    checkpoint = args.out_dir / "diagnostic_head.pth"
    torch.save({"architecture": ARCHITECTURE, "head_state": head.state_dict(),
                "kind": "tiny_train_panel_diagnostic_not_completed_pilot",
                "base_checkpoint_sha256": proof["artifacts_sha256"]["shared_velocity_e10.pth"],
                "generator_training_eligible": "hold_not_qualified"}, checkpoint)
    restored = ResidualVelocityHead()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["head_state"], strict=True)
    replayed = evaluate(restored, samples)
    if after != replayed:
        raise ValueError("saved head roundtrip did not reproduce CPU metrics")
    report = {
        "kind": "image_conditioned_residual_tiny_train_panel_diagnostic",
        "generator_training_eligible": "hold_not_qualified",
        "architecture": ARCHITECTURE,
        "source_sha256": sources, "pilot_artifacts_sha256": proof["artifacts_sha256"],
        "checkpoint_sha256": file_sha256(checkpoint),
        "fit_split": "train", "validation_or_test_access": "none",
        "annotations_enter_prediction": False, "annotations_enter_fit_loss": True,
        "base_predictor_updates": 0, "head_optimizer_steps": len(trace),
        "head_changed_tensors": sum(not torch.equal(initial[k], v)
                                    for k, v in head.state_dict().items()),
        "settings": {"seed": 0, "steps": args.steps, "lr": args.lr, "batch_size": 8,
                     "loss": "unchanged_v7_geometry", "best_iterate_selection": False,
                     "selection": "first4_fit_next4_probe_per_sorted_saved_pass_stratum"},
        "samples": [{k: s[k] for k in ("sequence_id", "role", "saved_stratum", "sample")}
                    for s in samples],
        "before": before, "after": after, "trace": trace,
        "summary": {role: {"observations": sum(r["role"] == role for r in before),
                           "before_pass": sum(joint(r) for r in before if r["role"] == role),
                           "after_pass": sum(joint(r) for r in after if r["role"] == role)}
                    for role in ("fit", "probe")},
        "runtime": {"device": "cpu", "seconds": time.monotonic()-started,
                    "torch": str(torch.__version__)},
        "limitations": ["Selected train panel, not a dataset pass-rate estimate.",
                        "Probe withheld only from head optimizer; v7 saw these train sequences.",
                        "No independent fine-grained correspondence evidence.",
                        "Not a completed pilot or final300-epoch experiment."],
    }
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
