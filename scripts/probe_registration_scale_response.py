"""CPU source-scale equivariance diagnostic, not physical correspondence GT.

Scale RGB about its annotated target centre, leave IR unchanged, and compare
predictions with the known transform of each model's own original sampling map.
Annotations define perturbation/measurement only; models still see images only.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import (
    box_points,
    centre_grid,
    map_boxes,
    map_points,
    sampling_map,
    valid_support,
)
from aero_ir.utils.manifest import file_sha256
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.render_registration_v7 import selected_cases
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import ARMS, load_checkpoint

SCALES = ((1., 1.), (1.25, 1.25), (.8, .8), (1.25, 1.), (.8, 1.),
          (1., 1.25), (1., .8))


def affine(points, centre, sx, sy, *, inverse=False):
    """Centre is normalized [-1,1] xy; positive scales preserve axis orientation."""
    if not all(math.isfinite(s) and s > 0 for s in (sx, sy)):
        raise ValueError("finite positive scales required")
    if centre.shape != (2,) or not torch.isfinite(centre).all():
        raise ValueError("one finite xy centre required")
    scale = points.new_tensor([sx, sy])
    return centre + ((points - centre) / scale if inverse else (points - centre) * scale)


def scaled_image(image, centre, sx, sy):
    """Forward content scaling Q uses inverse Q for raster pullback sampling."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("one RGB image required")
    grid = centre_grid(image[:, :2])
    return F.grid_sample(image, affine(grid, centre, sx, sy, inverse=True),
                         align_corners=False, padding_mode="zeros")


def scale_response(base, prediction, target_box, centre, sx, sy):
    if base.shape != prediction.shape or base.shape[:2] != (1, 2):
        raise ValueError("one matching pair of two-channel fields required")
    if not torch.isfinite(base).all():
        raise ValueError("nonfinite reference field")
    h, w = base.shape[-2:]
    reference = sampling_map(base)
    expected = affine(reference, centre, sx, sy)
    # Never let the new prediction choose which pixels get scored.
    support = valid_support(reference, h, w) & valid_support(expected, h, w)
    grid = centre_grid(base)
    roi = (((grid + 1) / 2 - target_box[:, None, None, :2]).abs()
           <= target_box[:, None, None, 2:] / 2).all(-1)
    finite = bool(torch.isfinite(prediction).all())
    error = torch.linalg.vector_norm(
        (sampling_map(prediction) - expected) * base.new_tensor([w/2, h/2]), dim=-1)
    result = {"finite_prediction": finite}
    for name, mask in (("global", support), ("target_roi", support & roi)):
        count = int(mask.sum())
        result[name] = {
            "observed_pixels": count,
            "median_error_pixels": float(error[mask].median()) if count and finite else None,
            "p95_error_pixels": float(torch.quantile(error[mask], .95))
            if count and finite else None,
        }
    points = box_points(target_box)
    mapped = map_points(base, points)
    supported = bool((valid_support(points, h, w) & valid_support(mapped, h, w)
                      & valid_support(affine(mapped, centre, sx, sy), h, w)).all())
    old_size = map_boxes(target_box, base)[0, 2:]
    new_size = map_boxes(target_box, prediction)[0, 2:]
    usable = supported and finite and bool((old_size > 1e-7).all())
    ratios = (new_size / old_size).tolist() if usable else [None, None]
    gains = [math.log(r) / math.log(s) if r is not None and r > 0 and s != 1 else None
             for r, s in zip(ratios, (sx, sy), strict=True)]
    result["mapped_target_box"] = {
        "reference_and_expected_supported": supported,
        "size_ratio_xy": ratios, "log_scale_gain_xy": gains,
    }
    return result


def median(values):
    finite = [x for x in values if x is not None]
    return statistics.median(finite) if finite else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path,
                   default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    p.add_argument("--run-root", type=Path,
                   default=Path("experiments/registration_residual_pilot_e10_seed0"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("refusing to overwrite scale diagnostic")
    hashes = pilot_hashes(args.run_root)
    compare(args.run_root, args.cache_root)
    if pilot_hashes(args.run_root) != hashes:
        raise ValueError("pilot changed during verification")
    sources = audit_sources()
    for name in ("scripts/probe_registration_scale_response.py",
                 "scripts/render_registration_v7.py"):
        sources[name] = file_sha256(name)
    screens = [json.loads((args.run_root / arm / "final_train_screen.json").read_text())
               for arm in ARMS]
    groups, selected = selected_cases(*(s["rows"] for s in screens))
    samples = {r["sequence_id"]: s for r, s in zip(screens[0]["rows"], screens[0]["samples"],
                                                strict=True)}
    torch.set_num_threads(1)
    device = torch.device("cpu")
    models = {a: load_checkpoint(args.run_root / a / "final.pth", device)[0] for a in ARMS}
    rows = []
    with torch.no_grad():
        for group, sid in selected:
            sample = samples[sid]
            arrays, digest = read_batch(args.cache_root, sample["selection"])
            if digest != sample["array_sha256"]:
                raise ValueError("cached observation differs")
            visible, infrared, vb, ib = tensors(arrays, device)
            centre = 2 * vb[0, :2] - 1
            for arm, model in models.items():
                base, _ = model.fields(infrared, visible)
                for sx, sy in SCALES:
                    modified = scaled_image(visible, centre, sx, sy)
                    prediction, _ = model.fields(infrared, modified)
                    rows.append({"sequence_id": sid, "stratum": group, "arm": arm,
                                 "sample": sample, "source_scale_xy": [sx, sy],
                                 "source_centre_normalized_xy": centre.tolist(),
                                 "response": scale_response(base, prediction, ib, centre, sx, sy)})
            print(f"tested source scales: {sid} ({group})", flush=True)
    summary = {}
    for arm in ARMS:
        summary[arm] = []
        for scale in SCALES:
            values = [r["response"] for r in rows if r["arm"] == arm
                      and r["source_scale_xy"] == list(scale)]
            errors = [v["target_roi"]["median_error_pixels"] for v in values]
            box_values = [v["mapped_target_box"] for v in values]
            summary[arm].append({
                "scale_xy": scale, "observations": len(values),
                "measured_rois": sum(e is not None for e in errors),
                "median_of_roi_median_error_pixels": median(errors),
                "measured_box_ratios": sum(v["size_ratio_xy"][0] is not None for v in box_values),
                "median_size_ratio_xy": [median([v["size_ratio_xy"][i] for v in box_values])
                                         for i in range(2)],
                "median_log_scale_gain_xy": [median([v["log_scale_gain_xy"][i]
                                                       for v in box_values]) for i in range(2)],
            })
    if pilot_hashes(args.run_root) != hashes:
        raise ValueError("pilot drift during diagnostic")
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("source drift during diagnostic")
    report = {
        "kind": "source_scale_response_not_correspondence_accuracy",
        "evaluated_split": "train", "validation_or_test_access": "none",
        "source_sha256": sources, "pilot_artifacts_sha256": hashes,
        "selection": "first two sorted IDs per paired final-pass stratum",
        "stratum_populations": {k: len(v) for k, v in groups.items()},
        "scales_xy": SCALES, "summary": summary, "rows": rows,
        "parameter_updates": 0, "device": "cpu",
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "Expected maps derive from potentially incorrect original model predictions.",
            "Train boxes set source augmentation centre and target measurement, not model input.",
            "Eight stratified train cases, not representative or held-out accuracy.",
            "Scale affects image context/HUD and bilinear raster interpolation as well as target.",
            "Forward response only; no independent physical or reverse-correspondence evidence.",
            "Gain1 is ideal; gain0 is unchanged. Identity scale has undefined log gain.",
            "Fixed reference support cannot be reduced by a bad new prediction.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
