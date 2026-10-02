"""Check scale-response gradients in the saved residual head on CPU; no fitting."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from aero_ir.registration.protocol_v7 import training_loss
from aero_ir.registration.scale_equivariance import source_scale_loss
from aero_ir.utils.manifest import file_sha256
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.probe_registration_scale_response import SCALES, scaled_image
from scripts.render_registration_v7 import selected_cases
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import ARMS, load_checkpoint

WEIGHT = .1  # Explicit candidate diagnostic weight, not chosen by gate-pass search.


def gradient(loss, parameters):
    values = torch.autograd.grad(loss, parameters)
    vector = torch.cat([v.flatten() for v in values])
    if not torch.isfinite(vector).all():
        raise FloatingPointError("nonfinite head gradient")
    return vector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path,
                   default=Path("experiments/registration_residual_pilot_e10_seed0"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("refusing to overwrite gradient diagnostic")
    hashes = pilot_hashes(args.run_root)
    compare(args.run_root, args.cache_root)
    sources = audit_sources()
    for name in ("scripts/probe_registration_scale_gradient.py",
                 "scripts/probe_registration_scale_response.py",
                 "scripts/render_registration_v7.py"):
        sources[name] = file_sha256(name)
    screens = [json.loads((args.run_root / arm / "final_train_screen.json").read_text())
               for arm in ARMS]
    _, selected = selected_cases(*(s["rows"] for s in screens))
    samples = {r["sequence_id"]: s for r, s in zip(screens[0]["rows"], screens[0]["samples"],
                                                strict=True)}
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model, _ = load_checkpoint(args.run_root / "residual_head" / "final.pth", device)
    parameters = list(model.head.parameters())
    rows = []
    for group, sid in selected:
        sample = samples[sid]
        arrays, digest = read_batch(args.cache_root, sample["selection"])
        if digest != sample["array_sha256"]:
            raise ValueError("cache bytes differ")
        visible, infrared, vb, ib = tensors(arrays, device)
        fields = model.fields(infrared, visible)
        reference = fields[0].detach()
        geometry, _ = training_loss(visible, infrared, vb, ib, fields)
        original_gradient = gradient(geometry, parameters)
        centre = 2 * vb[0, :2] - 1
        for sx, sy in SCALES[1:]:
            prediction, _ = model.fields(infrared, scaled_image(visible, centre, sx, sy))
            loss, support = source_scale_loss(reference, prediction, ib, centre[None],
                                               reference.new_tensor([[sx, sy]]))
            scale_gradient = gradient(loss, parameters)
            a, b = float(original_gradient.norm()), float(scale_gradient.norm())
            rows.append({"sequence_id": sid, "stratum": group, "sample": sample,
                         "scale_xy": [sx, sy], "geometry_loss": float(geometry.detach()),
                         "scale_loss": float(loss.detach()), "support": support,
                         "geometry_gradient_norm": a, "scale_gradient_norm": b,
                         "weighted_gradient_norm_ratio": WEIGHT*b/a if a else None,
                         "gradient_cosine": float(original_gradient @ scale_gradient)/(a*b)
                         if a*b else None})
        print(f"checked scale gradients: {sid}", flush=True)
    if pilot_hashes(args.run_root) != hashes:
        raise ValueError("pilot artifact drift")
    if any(file_sha256(name) != digest for name, digest in sources.items()):
        raise ValueError("source drift")
    cosines = [r["gradient_cosine"] for r in rows if r["gradient_cosine"] is not None]
    ratios = [r["weighted_gradient_norm_ratio"] for r in rows
              if r["weighted_gradient_norm_ratio"] is not None]
    summary = {"observations": len(rows), "median_gradient_cosine": statistics.median(cosines),
               "positive_cosines": sum(c > 0 for c in cosines),
               "negative_cosines": sum(c < 0 for c in cosines),
               "median_weighted_gradient_norm_ratio": statistics.median(ratios)}
    result = {"kind": "scale_equivariance_gradient_not_qualification", "device": "cpu",
              "arm": "residual_head", "parameter_updates": 0, "evaluated_split": "train",
              "validation_or_test_access": "none", "candidate_weight": WEIGHT,
              "source_sha256": sources, "pilot_artifacts_sha256": hashes,
              "summary": summary, "rows": rows,
              "generator_training_eligible": "hold_not_qualified",
              "limitations": ["Gradient viability is not achieved repair or generalization.",
                              "Original maps are detached pseudo-targets, not physical GT.",
                              "Use with geometric supervision; equivariance alone can collapse.",
                              "Eight previously selected train frames; no optimizer updates."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
