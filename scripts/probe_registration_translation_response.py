"""Known source-translation response of completed pilots on eight train cases.

No fitting. Synthetic shifts test geometric equivariance, not true cross-modal
correspondence. Expected forward maps use the model's own unshifted prediction.
"""

import argparse
import json
from pathlib import Path

import torch

from aero_ir.registration.geometry import centre_grid, sampling_map, valid_support
from aero_ir.utils.manifest import file_sha256
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.render_registration_v7 import selected_cases
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import ARMS, load_checkpoint

SHIFTS = ((4, 0), (-4, 0), (0, 4), (0, -4))


def shifted_image(image, dx, dy):
    """Exact integer pixel transport, with zeros only in newly uncovered borders."""
    h, w = image.shape[-2:]
    if type(dx) is not int or type(dy) is not int or abs(dx) >= w or abs(dy) >= h:
        raise ValueError("require bounded integer translation")
    shifted = torch.roll(image, shifts=(dy, dx), dims=(-2, -1)).clone()
    if dx > 0:
        shifted[..., :dx] = 0
    elif dx < 0:
        shifted[..., dx:] = 0
    if dy > 0:
        shifted[..., :dy, :] = 0
    elif dy < 0:
        shifted[..., dy:, :] = 0
    return shifted


def response(base, prediction, target_box, dx, dy):
    """Centre fields: source shifted by+d requires forward pullback shifted by+d."""
    if base.shape != prediction.shape or base.shape[:2] != (1, 2) or not (dx or dy):
        raise ValueError("one matching pair of two-channel fields and nonzero shift required")
    if not torch.isfinite(base).all():
        raise ValueError("nonfinite unshifted reference field")
    h, w = base.shape[-2:]
    delta = base.new_tensor([2 * dx / w, 2 * dy / h])[None, :, None, None]
    expected = base + delta
    # Fixed support is defined by the reference and known translation, never by
    # the new prediction; a bad prediction cannot mask its own errors away.
    support = valid_support(sampling_map(base), h, w)
    support &= valid_support(sampling_map(expected), h, w)
    positions = (centre_grid(base) + 1) / 2
    roi = ((positions - target_box[:, None, None, :2]).abs()
           <= target_box[:, None, None, 2:] / 2).all(-1)
    finite = bool(torch.isfinite(prediction).all())
    error = (prediction - expected) * base.new_tensor([w/2, h/2])[None, :, None, None]
    norm = torch.linalg.vector_norm(error, dim=1)
    movement = (prediction - base) * base.new_tensor([w/2, h/2])[None, :, None, None]
    gain = (movement[:, 0] * dx + movement[:, 1] * dy) / (dx*dx + dy*dy)
    result = {"finite_prediction": finite}
    for name, mask in (("global", support), ("target_roi", support & roi)):
        count = int(mask.sum())
        result[name] = {
            "observed_pixels": count,
            "median_error_pixels": float(norm[mask].median()) if count and finite else None,
            "p95_error_pixels": float(torch.quantile(norm[mask], .95))
            if count and finite else None,
            "mean_projected_response_gain": float(gain[mask].mean())
            if count and finite else None,
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path,
                   default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    p.add_argument("--run-root", type=Path,
                   default=Path("experiments/registration_residual_pilot_e10_seed0"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("refusing to overwrite response diagnostic")
    hashes = pilot_hashes(args.run_root)
    compare(args.run_root, args.cache_root)
    if pilot_hashes(args.run_root) != hashes:
        raise ValueError("pilot changed during verification")
    sources = audit_sources()
    for name in ("scripts/probe_registration_translation_response.py",
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
            visible, infrared, _, ib = tensors(arrays, device)
            for arm, model in models.items():
                base, _ = model.fields(infrared, visible)
                for dx, dy in SHIFTS:
                    prediction, _ = model.fields(infrared, shifted_image(visible, dx, dy))
                    rows.append({"sequence_id": sid, "stratum": group, "arm": arm,
                                 "sample": sample, "source_shift_pixels_xy": [dx, dy],
                                 "response": response(base, prediction, ib, dx, dy)})
            print(f"tested source shifts: {sid} ({group})", flush=True)
    summary = {}
    for arm in ARMS:
        values = [r["response"]["target_roi"] for r in rows if r["arm"] == arm]
        measured = [v for v in values if v["median_error_pixels"] is not None]
        summary[arm] = {
            "shifted_observations": len(values), "measured_target_rois": len(measured),
            "median_of_roi_median_error_pixels": float(torch.tensor(
                [v["median_error_pixels"] for v in measured]).quantile(.5)) if measured else None,
            "median_of_roi_mean_response_gain": float(torch.tensor(
                [v["mean_projected_response_gain"] for v in measured]).quantile(.5))
            if measured else None,
        }
    if pilot_hashes(args.run_root) != hashes:
        raise ValueError("pilot drift during diagnostic")
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("source drift during diagnostic")
    report = {
        "kind": "source_translation_response_not_correspondence_accuracy",
        "evaluated_split": "train", "validation_or_test_access": "none",
        "source_sha256": sources, "pilot_artifacts_sha256": hashes,
        "selection": "first two sorted IDs per paired final-pass stratum",
        "stratum_populations": {k: len(v) for k, v in groups.items()}, "shifts": SHIFTS,
        "summary": summary, "rows": rows, "parameter_updates": 0, "device": "cpu",
        "expected_response_gain": 1, "unchanged_prediction_response_gain": 0,
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "Expected translation is relative to the model's own potentially incorrect base map.",
            "Forward-field response only; no independent reverse correspondence evidence.",
            "Synthetic source translation also moves HUD and introduces zero border context.",
            "Support is fixed from baseline and known transform, not new prediction.",
            "Eight stratified training frames, not representative/held-out performance.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
