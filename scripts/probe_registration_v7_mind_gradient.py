"""Train-only diagnostic of MIND's local gradient at a completed v7 mapping.

No fitting or parameter updates. Gradients are with respect to the shared velocity,
not network parameters. Their magnitude/alignment cannot prove physical accuracy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aero_ir.registration.geometry import from_superfusion
from aero_ir.registration.protocol_v7 import (
    INTEGRATION_STEPS,
    MIND_WEIGHT,
    descriptor_direction,
    observation,
    training_loss,
)
from aero_ir.registration.shared_velocity import shared_fields
from aero_ir.utils.manifest import file_sha256
from scripts.train_antiuav300_registration_v7 import (
    automatic_exclusion,
    load_trained,
    read_batch,
    tensors,
)
from scripts.verify_registration_v7 import verify_arm


def gradient_terms(velocity, visible, infrared, vb, ib, observations):
    # Detach from the predictor deliberately: measure the local deformation-space
    # objective, without consuming memory for the encoder's parameter Jacobian.
    velocity = velocity.detach().clone().requires_grad_(True)
    fields = shared_fields(velocity, INTEGRATION_STEPS)
    geometry, _ = training_loss(visible, infrared, vb, ib, fields)
    a, da = descriptor_direction(observations[0], observations[1], fields[0])
    b, db = descriptor_direction(observations[1], observations[0], fields[1])
    mind = (a + b) / 2
    g = torch.autograd.grad(geometry, velocity, retain_graph=True)[0].flatten()
    m = torch.autograd.grad(MIND_WEIGHT * mind, velocity)[0].flatten()
    if not all(torch.isfinite(v).all() for v in (g, m, geometry, mind)):
        raise ValueError("nonfinite diagnostic objective or velocity gradient")
    gn, mn = float(g.norm()), float(m.norm())
    return {
        "geometry_loss": float(geometry.detach()),
        "raw_mind_loss": float(mind.detach()),
        "mind_weight": MIND_WEIGHT,
        "geometry_gradient_norm": gn,
        "weighted_mind_gradient_norm": mn,
        "weighted_mind_to_geometry_norm_ratio": mn / gn if gn > 0 else None,
        "gradient_cosine": float(torch.dot(g, m)) / (gn * mn) if gn > 0 and mn > 0 else None,
        "mind_target_eligible_fraction": (da["eligible_fraction"] + db["eligible_fraction"]) / 2,
        "mind_source_observed_fraction": (
            da["source_observed_fraction"] + db["source_observed_fraction"]
        )
        / 2,
    }


def summarize(rows):
    summary = {}
    for key in ("weighted_mind_to_geometry_norm_ratio", "gradient_cosine"):
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = {
            "measured": len(values),
            "undefined": len(rows) - len(values),
            "median": float(np.median(values)) if values else None,
            "p05": float(np.quantile(values, 0.05)) if values else None,
            "p95": float(np.quantile(values, 0.95)) if values else None,
        }
    summary["positive_cosine_frames"] = sum(
        r["gradient_cosine"] is not None and r["gradient_cosine"] > 0 for r in rows
    )
    summary["negative_cosine_frames"] = sum(
        r["gradient_cosine"] is not None and r["gradient_cosine"] < 0 for r in rows
    )
    summary["zero_mind_gradient_frames"] = sum(r["weighted_mind_gradient_norm"] == 0 for r in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v7_pilot_e10_seed0"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    parser.add_argument("--arm", choices=("geometry", "geometry_mind"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite gradient diagnostic")
    root = args.run_root / args.arm
    proof = verify_arm(root, args.arm, args.cache_root)
    if proof["status"] != "verified_saved_artifacts_not_replayed":
        parser.error("requires a verified completed pilot")
    project = Path(__file__).resolve().parents[1]
    source_hashes = {
        **proof["spec"]["source_sha256"],
        "scripts/probe_registration_v7_mind_gradient.py": file_sha256(__file__),
        "scripts/verify_registration_v7.py": file_sha256(
            Path(__file__).with_name("verify_registration_v7.py")
        ),
    }
    torch.set_num_threads(1)
    model, _ = load_trained(root / "shared_velocity_e10.pth", torch.device("cpu"))
    screen = json.loads((root / "final_train_screen.json").read_text())
    rows = []
    for row, sample in zip(screen["rows"], screen["samples"], strict=True):
        arrays, digest = read_batch(args.cache_root, sample["selection"])
        if digest != sample["array_sha256"]:
            raise ValueError("cached input differs from the completed train screen")
        visible, infrared, vb, ib = tensors(arrays, torch.device("cpu"))
        with torch.no_grad():
            velocity = from_superfusion(
                model.predictor(infrared, visible, direction="visible_to_infrared")
            )
            observations = [
                observation(
                    image, boxes, torch.from_numpy(automatic_exclusion(arrays[key][0]))[None]
                )
                for image, boxes, key in ((visible, vb, "visible"), (infrared, ib, "infrared"))
            ]
        measurements = gradient_terms(velocity, visible, infrared, vb, ib, observations)
        rows.append(
            {
                "sequence_id": row["sequence_id"],
                "selection": sample["selection"],
                "array_sha256": digest,
                **measurements,
            }
        )
        if len(rows) % 20 == 0:
            print(f"measured {len(rows)}/{len(screen['rows'])} train observations", flush=True)
    for name, digest in proof["artifacts_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("pilot changed during gradient diagnostic")
    if any(file_sha256(project / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("source changed during gradient diagnostic")
    report = {
        "kind": "v7_velocity_gradient_diagnostic_not_qualification",
        "arm": args.arm,
        "device": "cpu",
        "torch": str(torch.__version__),
        "model_mode": "eval",
        "optimizer_steps": 0,
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "source_sha256": source_hashes,
        "pilot_artifacts_sha256": proof["artifacts_sha256"],
        "gradient_domain": "shared stationary velocity; not predictor parameters",
        "limitations": [
            "Local first-order diagnostic, not an experiment with a different weight.",
            "Positive gradient cosine does not establish physical correspondence.",
            "No claim about parameter-space gradients or Adam/clipping behaviour.",
            "Undefined cosines/ratios remain null, never fabricated as zero.",
        ],
        "rows": rows,
        "summary": summarize(rows),
        "generator_training_eligible": "hold_not_qualified",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
