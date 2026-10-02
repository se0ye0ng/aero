"""Automatic SAM pseudo-silhouette alignment, with no human review prerequisite.

Box-assisted train-only hypothesis screen, not a registration checkpoint trainer.
Masks guide an invertible similarity transform; an independent prompt perturbation
checks mask/transform stability. Neither mask agreement nor analytic invertibility
is physical pixel GT. Paired and shuffled inputs use the exact same procedure.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from aero_ir.utils.manifest import file_sha256
from aero_ir.utils.paths import antiuav300_root
from scripts.probe_registration_auto_roi import load_native, prepare


def mask_iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else None


def perturbed_boxes(box, side):
    box = np.array(box, float)
    delta = max(2.0, 0.05 * float((box[2:] - box[:2]).max()))
    return [
        box,
        np.clip(box + [-delta, -delta, delta, delta], 0, side - 1),
        np.clip(box + [delta, delta, delta, delta], 0, side - 1),
    ]


def extract_masks(predictor, data):
    gray = data["inpaint"]
    predictor.set_image(np.repeat(gray[..., None], 3, axis=2))
    masks, confidences = [], []
    for box in perturbed_boxes(data["box"], len(gray)):
        proposed, quality, _ = predictor.predict(box=box, multimask_output=False)
        masks.append(proposed[0].astype(bool))
        confidences.append(float(quality[0]))
    a, b, c = masks
    reasons = []
    for mask in masks:
        if mask.sum() < 16:
            reasons.append("empty_or_tiny_mask")
        elif mask.mean() > 0.5:
            reasons.append("mask_covers_more_than_half_context")
        if mask[[0, -1]].any() or mask[:, [0, -1]].any():
            reasons.append("mask_touches_crop_boundary")
        if (mask & data["excluded"]).any():
            reasons.append("mask_intersects_excluded_hud_support")
    stability = [mask_iou(a, b), mask_iou(a, c)]
    if any(s is None or s < 0.8 for s in stability):
        reasons.append("prompt_jitter_iou_below_0_8")
    return masks, {
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "prompt_stability_ious": stability,
        "sam_self_predicted_iou_not_accuracy": confidences,
        "mask_pixels": [int(m.sum()) for m in masks],
    }


def matrices(side):
    """RGB-crop point -> IR-crop point, with analytic inverse, no reflections."""
    specs, values = [], []
    for scale in (0.9, 1.0, 1.1):
        for angle in (-10.0, 0.0, 10.0):
            for dx in (-8, -4, 0, 4, 8):
                for dy in (-8, -4, 0, 4, 8):
                    matrix = cv2.getRotationMatrix2D(((side - 1) / 2, (side - 1) / 2), angle, scale)
                    matrix[:, 2] += [dx, dy]
                    values.append(np.vstack((matrix, [0, 0, 1])))
                    specs.append({"scale": scale, "angle_degrees": angle, "dx": dx, "dy": dy})
    baseline = specs.index({"scale": 1.0, "angle_degrees": 0.0, "dx": 0, "dy": 0})
    return values, specs, baseline


def compare_masks(source, target, matrix):
    """Symmetric forward/inverse foreground-safe Dice and contour distance."""
    results = []
    for a, b, h in ((source, target, matrix), (target, source, np.linalg.inv(matrix))):
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        warped = cv2.warpPerspective(
            a.astype(np.uint8), h, b.shape[::-1], flags=cv2.INTER_NEAREST
        ).astype(bool)
        support = cv2.warpPerspective(
            np.ones(a.shape, np.uint8), h, b.shape[::-1], flags=cv2.INTER_NEAREST
        ).astype(bool)
        safe = binary_erosion(support, border_value=0)
        # Both directions are checked so disappearing source foreground cannot win.
        if not warped.any() or not b.any() or (warped & ~safe).any() or (b & ~safe).any():
            return None
        edge_a = warped & ~binary_erosion(warped)
        edge_b = b & ~binary_erosion(b)
        dice = 1 - 2 * (warped & b).sum() / (warped.sum() + b.sum())
        chamfer = (
            distance_transform_edt(~edge_a)[edge_b].mean()
            + distance_transform_edt(~edge_b)[edge_a].mean()
        ) / 2
        results.append((dice, chamfer))
    dice, chamfer = np.mean(results, axis=0)
    return {"dice_loss": float(dice), "chamfer_model_px": float(chamfer)}


def align(source_masks, target_masks):
    values, specs, baseline = matrices(source_masks[0].shape[0])
    scores = [compare_masks(source_masks[0], target_masks[0], m) for m in values]
    if scores[baseline] is None:
        return {"status": "invalid_baseline_support"}
    # Predeclared equal-scale components: normalized Chamfer and Dice.
    costs = [
        s["dice_loss"] + s["chamfer_model_px"] / source_masks[0].shape[0]
        if s is not None
        else float("inf")
        for s in scores
    ]
    best = baseline if costs[baseline] <= min(costs) + 1e-10 else int(np.argmin(costs))
    checks = []
    for a, b in zip(source_masks[1:], target_masks[1:], strict=True):
        checks.append(
            {
                "baseline": compare_masks(a, b, values[baseline]),
                "selected": compare_masks(a, b, values[best]),
            }
        )
    return {
        "status": "aligned_pseudo_masks_only",
        "baseline": scores[baseline],
        "selected": scores[best],
        "selected_spec": specs[best],
        "candidate_count": len(values),
        "eligible_candidates": sum(s is not None for s in scores),
        "matrix_rgb_crop_to_ir_crop": values[best].tolist(),
        "inverse_matrix": np.linalg.inv(values[best]).tolist(),
        "determinant": float(np.linalg.det(values[best])),
        "heldout_prompt_checks_not_independent_gt": checks,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=antiuav300_root())
    p.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("experiments/detector_free_train16_01/report.json"),
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("choose fresh output directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    from segment_anything import SamPredictor, sam_model_registry

    if importlib.metadata.version("segment-anything") != "1.0":
        raise ValueError("requires pinned segment-anything==1.0")
    baseline = json.loads(args.baseline_report.read_text())
    if baseline["evaluated_split"] != "train" or baseline["validation_or_test_access"] != "none":
        raise ValueError("requires train-only panel")
    if file_sha256(args.root / "label_new/train.json") != baseline["split_manifest_sha256"]:
        raise ValueError("split changed")
    records = baseline["inputs"]
    if len(records) < 2 or len({r["sequence_id"] for r in records}) != len(records):
        raise ValueError("distinct sequence controls required")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    weight_hash = file_sha256(args.weights)
    model = sam_model_registry["vit_b"](checkpoint=None)
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True), strict=True
    )
    predictor = SamPredictor(model.eval().to(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, extracted, mask_rows, artifacts = time.monotonic(), [], [], {}
    with torch.inference_mode():
        for i, record in enumerate(records):
            pair = []
            for modality, (gray, box) in zip(
                ("rgb", "ir"), load_native(args.root, record), strict=True
            ):
                data = prepare(gray, box, 256)
                masks, stats = extract_masks(predictor, data)
                pair.append((masks, stats))
                path = args.output_dir / f"{i:03d}_{modality}.npz"
                np.savez_compressed(
                    path,
                    masks=np.stack(masks),
                    image=data["inpaint"],
                    excluded=data["excluded"],
                    prompt_box=data["box"],
                )
                artifacts[path.name] = file_sha256(path)
                mask_rows.append(
                    {
                        "sequence_id": record["sequence_id"],
                        "modality": modality,
                        "stats": stats,
                        "crop_metadata": data["meta"],
                        "artifact": path.name,
                    }
                )
            extracted.append(pair)
            print(f"{i + 1}/{len(records)} SAM masks and prompt-jitter controls", flush=True)
    rows = []
    for i, pair in enumerate(extracted):
        for kind, j in (("paired", i), ("shuffled", (i + 1) % len(extracted))):
            (a, sa), (b, sb) = pair[0], extracted[j][1]
            result = (
                align(a, b)
                if sa["eligible"] and sb["eligible"]
                else {"status": "unstable_or_invalid_masks"}
            )
            rows.append(
                {
                    "kind": kind,
                    "rgb_sequence": records[i]["sequence_id"],
                    "ir_sequence": records[j]["sequence_id"],
                    "result": result,
                }
            )
    # End-to-end geometry implementation control, not a control of SAM semantics.
    mask = np.zeros((64, 64), bool)
    mask[20:30, 24:35] = True
    mask[30:40, 24:28] = True
    shifted = np.roll(mask, 4, axis=1)
    control = align([mask] * 3, [shifted] * 3)
    if control["selected"]["dice_loss"] > 1e-8:
        raise RuntimeError("known-shift silhouette control failed")
    summary = {}
    for kind in ("paired", "shuffled"):
        group = [r["result"] for r in rows if r["kind"] == kind]
        scored = [r for r in group if r["status"] == "aligned_pseudo_masks_only"]
        summary[kind] = {
            "total": len(group),
            "eligible": len(scored),
            "median_selected_dice_loss": float(
                np.median([r["selected"]["dice_loss"] for r in scored])
            )
            if scored
            else None,
            "median_baseline_dice_loss": float(
                np.median([r["baseline"]["dice_loss"] for r in scored])
            )
            if scored
            else None,
        }
    project = Path(__file__).resolve().parents[1]
    sources = (
        "scripts/probe_registration_sam.py",
        "scripts/probe_registration_auto_roi.py",
        "scripts/audit_antiuav300_dense_registration.py",
    )
    import segment_anything

    package = Path(segment_anything.__file__).parent
    report = {
        "experiment": "automatic_sam_silhouette_similarity_v1",
        "summary": summary,
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "torch_version": torch.__version__,
        "sam_version": "1.0",
        "sam_source_sha256": {
            str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))
        },
        "weights_sha256": weight_hash,
        "baseline_sha256": file_sha256(args.baseline_report),
        "inputs": records,
        "mask_rows": mask_rows,
        "rows": rows,
        "artifacts_sha256": artifacts,
        "known_shift_control": control,
        "sources": {s: file_sha256(project / s) for s in sources},
        "protocol": {
            "human_input": False,
            "train_box_prompted": True,
            "rgb_input": "grayscale repeated to 3 channels to match the ROI screen",
            "masks_are_pseudo_labels_not_gt": True,
            "prompt_stability_iou_min": 0.8,
            "transformation": "shared orientation-preserving similarity + analytic inverse",
            "no_registration_or_generator_training": True,
            "selection": "Dice + symmetric Chamfer/256 on original prompts only",
            "scope": "target crops only; no full-image qualification",
            "perturbed_prompts_are_not_independent_gt": True,
        },
        "qualification": "not_assessed_pseudo_mask_diagnostic",
    }
    if file_sha256(args.weights) != weight_hash:
        raise RuntimeError("weights changed during run")
    with (args.output_dir / "report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
