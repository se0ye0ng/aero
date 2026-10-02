"""Repair SAM ROI coordinates and bounded similarity search, without human inputs.

replay: reproject existing masks, no SAM inference and no claim of new segmentation.
extract: rerun the same frozen SAM on corrected crops before the same CPU search.
Neither route grants dense/full-image registration qualification.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.roi_geometry import prepare_centred, restore_cached
from aero_ir.registration.silhouette_search import align_masks
from aero_ir.utils.manifest import file_sha256
from aero_ir.utils.paths import antiuav300_root
from scripts.probe_registration_auto_roi import auto_hud, load_native
from scripts.probe_registration_sam import extract_masks

WEIGHT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"


def read_previous(path):
    report = json.loads(path.read_text())
    if (
        report.get("experiment") != "automatic_sam_silhouette_similarity_v1"
        or report.get("evaluated_split") != "train"
        or report.get("validation_or_test_access") != "none"
    ):
        raise ValueError("requires original train-only SAM v1 report")
    for name, digest in report["artifacts_sha256"].items():
        artifact = (path.parent / name).resolve()
        if not artifact.is_relative_to(path.parent.resolve()) or file_sha256(artifact) != digest:
            raise ValueError(f"invalid previous artifact {name}")
    lookup = {(r["sequence_id"], r["modality"]): r for r in report["mask_rows"]}
    ids = [r["sequence_id"] for r in report["inputs"]]
    if (
        len(ids) < 2
        or len(set(ids)) != len(ids)
        or len(report["mask_rows"]) != 2 * len(ids)
        or set(lookup) != {(i, m) for i in ids for m in ("rgb", "ir")}
    ):
        raise ValueError("incomplete or duplicated original panel")
    return report, lookup


def summarize(rows):
    summary = {}
    for kind in ("paired", "shuffled"):
        group = [r for r in rows if r["kind"] == kind]
        fits = [r["result"] for r in group if r["result"]["status"] == "fit_pseudo_masks_only"]
        baselines = [r for r in fits if r["baseline"] is not None]
        summary[kind] = {
            "total": len(group),
            "fit": len(fits),
            "comparable_baseline_count": len(baselines),
            "median_dice_before": float(np.median([r["baseline"]["dice_loss"] for r in baselines]))
            if baselines
            else None,
            "median_dice_after_same_cases": float(
                np.median([r["selected"]["dice_loss"] for r in baselines])
            )
            if baselines
            else None,
            "median_chamfer_before": float(
                np.median([r["baseline"]["chamfer_model_px"] for r in baselines])
            )
            if baselines
            else None,
            "median_chamfer_after_same_cases": float(
                np.median([r["selected"]["chamfer_model_px"] for r in baselines])
            )
            if baselines
            else None,
            "dice_improved": sum(
                r["selected"]["dice_loss"] < r["baseline"]["dice_loss"] - 1e-10 for r in baselines
            ),
            "hard_boundary": sum(r["at_hard_search_boundary"] for r in fits),
            "near_explored_translation_edge": sum(
                r["near_explored_translation_edge"] for r in fits
            ),
            "near_explored_angle_edge": sum(r["near_explored_angle_edge"] for r in fits),
            "outside_initial_scale_grid": sum(r["outside_initial_scale_grid"] for r in fits),
            "chamfer_worsened_from_baseline": sum(
                r["chamfer_worsened_from_baseline"] for r in fits
            ),
        }
    paired = {
        r["rgb_sequence"]: r["result"]
        for r in rows
        if r["kind"] == "paired" and r["result"]["status"] == "fit_pseudo_masks_only"
    }
    shuffled = {
        r["rgb_sequence"]: r["result"]
        for r in rows
        if r["kind"] == "shuffled" and r["result"]["status"] == "fit_pseudo_masks_only"
    }
    common = paired.keys() & shuffled.keys()
    summary["common_source_comparison"] = {
        "count": len(common),
        "paired_lower_dice": sum(
            paired[k]["selected"]["dice_loss"] < shuffled[k]["selected"]["dice_loss"] - 1e-10
            for k in common
        ),
        "equal_dice": sum(
            abs(paired[k]["selected"]["dice_loss"] - shuffled[k]["selected"]["dice_loss"]) <= 1e-10
            for k in common
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "extract"), default="replay")
    parser.add_argument(
        "--previous-report",
        type=Path,
        default=Path("experiments/registration_sam_train16_v1/report.json"),
    )
    parser.add_argument(
        "--root", type=Path, default=antiuav300_root()
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("experiments/detector_free_train16_01/report.json"),
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("experiments/external/sam_vit_b_01ec64.pth")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose fresh output directory; never overwrite v1 or prior repairs")
    if args.mode == "replay" and args.device != "cpu":
        parser.error("cached-mask repair is CPU-only; use --mode extract for SAM inference")
    previous, lookup = read_previous(args.previous_report)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    predictor = None
    if args.mode == "extract":
        import segment_anything
        from segment_anything import SamPredictor, sam_model_registry

        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA unavailable")
        if importlib.metadata.version("segment-anything") != "1.0":
            raise ValueError("requires segment-anything==1.0")
        package = Path(segment_anything.__file__).parent
        installed_sources = {
            str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))
        }
        if installed_sources != previous["sam_source_sha256"]:
            raise ValueError("SAM package changed from v1")
        if (
            file_sha256(args.weights) != WEIGHT_SHA256
            or previous["weights_sha256"] != WEIGHT_SHA256
        ):
            raise ValueError("SAM weights differ from completed v1")
        if file_sha256(args.baseline_report) != previous["baseline_sha256"]:
            raise ValueError("fixed panel baseline changed")
        baseline = json.loads(args.baseline_report.read_text())
        if (
            previous["inputs"] != baseline["inputs"]
            or file_sha256(args.root / "label_new/train.json") != baseline["split_manifest_sha256"]
        ):
            raise ValueError("train inputs changed")
        model = sam_model_registry["vit_b"](checkpoint=None)
        model.load_state_dict(
            torch.load(args.weights, map_location="cpu", weights_only=True), strict=True
        )
        predictor = SamPredictor(model.eval().to(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, pairs, mask_rows, artifacts = time.monotonic(), [], [], {}
    with torch.inference_mode():
        for i, record in enumerate(previous["inputs"]):
            native = load_native(args.root, record) if predictor is not None else None
            pair = []
            for m, modality in enumerate(("rgb", "ir")):
                old = lookup[(record["sequence_id"], modality)]
                if predictor is None:
                    with np.load(
                        args.previous_report.parent / old["artifact"], allow_pickle=False
                    ) as z:
                        data = restore_cached(
                            z["masks"],
                            z["image"],
                            z["excluded"],
                            z["prompt_box"],
                            old["crop_metadata"],
                        )
                    # Preserve original abstentions, never use post-hoc outcomes to recover them.
                    stats = old["stats"]
                else:
                    gray, box = native[m]
                    data = prepare_centred(gray, box, auto_hud(gray))
                    masks, stats = extract_masks(predictor, data)
                    data["masks"] = np.stack(masks)
                    data["image"] = data["inpaint"]
                path = args.output_dir / f"{i:03d}_{modality}.npz"
                np.savez_compressed(
                    path,
                    masks=data["masks"],
                    image=data["image"],
                    valid=data["valid"],
                    excluded=data["excluded"],
                    prompt_box=data["box"],
                )
                artifacts[path.name] = file_sha256(path)
                mask_rows.append(
                    {
                        "sequence_id": record["sequence_id"],
                        "modality": modality,
                        "stats": stats,
                        "original_stats": old["stats"],
                        "crop_metadata": data["meta"],
                        "artifact": path.name,
                    }
                )
                data["stats"] = stats
                pair.append(data)
            pairs.append(pair)
            label = "reprojected cached" if predictor is None else "new SAM"
            print(
                f"{i + 1}/{len(previous['inputs'])} {label} masks",
                flush=True,
            )
    rows = []
    for i, pair in enumerate(pairs):
        for kind, j in (("paired", i), ("shuffled", (i + 1) % len(pairs))):
            source, target = pair[0], pairs[j][1]
            result = (
                align_masks(source, target)
                if source["stats"]["eligible"] and target["stats"]["eligible"]
                else {"status": "unstable_or_invalid_masks"}
            )
            rows.append(
                {
                    "kind": kind,
                    "rgb_sequence": previous["inputs"][i]["sequence_id"],
                    "ir_sequence": previous["inputs"][j]["sequence_id"],
                    "result": result,
                }
            )
        print(f"{i + 1}/{len(pairs)} corrected-coordinate paired/shuffled searches", flush=True)
    project = Path(__file__).resolve().parents[1]
    source_names = (
        "scripts/probe_registration_sam_v2.py",
        "src/aero_ir/registration/roi_geometry.py",
        "src/aero_ir/registration/silhouette_search.py",
        "scripts/probe_registration_sam.py",
        "scripts/probe_registration_auto_roi.py",
        "scripts/audit_antiuav300_dense_registration.py",
    )
    # Preserve exact executed sources, even if working files change after this run.
    for name in source_names:
        snapshot = args.output_dir / "sources" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, snapshot)
        artifacts[str(snapshot.relative_to(args.output_dir))] = file_sha256(snapshot)
    report = {
        "experiment": "sam_centred_roi_similarity_v2",
        "mode": args.mode,
        "device": args.device,
        "torch_version": torch.__version__,
        "previous_report_sha256": file_sha256(args.previous_report),
        "inputs": previous["inputs"],
        "weights_sha256": previous["weights_sha256"],
        "sam_source_sha256": previous["sam_source_sha256"] if predictor is None else {},
        "summary": summarize(rows),
        "rows": rows,
        "mask_rows": mask_rows,
        "artifacts_sha256": artifacts,
        "sources": {s: file_sha256(project / s) for s in source_names},
        "elapsed_seconds": time.monotonic() - started,
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "human_input": False,
        "protocol": {
            "existing_train_boxes_used": True,
            "native_roi_isotropic": True,
            "padding_is_invalid_support": True,
            "crop_side": 256,
            "sam_reextracted": predictor is not None,
            "replay_keeps_original_eligibility": predictor is None,
            "mask_stability_threshold": 0.8,
            "coarse_grid": {
                "scale": [0.6, 0.8, 1, 1.25, 1.667],
                "angle": [-30, -15, 0, 15, 30],
                "xy": [-32, -16, 0, 16, 32],
            },
            "beam_width": 2,
            "refinement": [[128, 1.15, 10, 8], [256, 1.05, 3, 2], [256, 1.02, 1, 1]],
            "hard_limits": {"scale": [0.5, 2], "angle": [-45, 45], "xy": [-64, 64]},
            "same_search_for_paired_and_shuffled": True,
            "objective": "symmetric Dice loss + Euclidean Chamfer / crop side",
            "scope": "target ROI only, no generator training",
            "perturbed_prompts_are_not_independent_gt": True,
        },
        "qualification": "not_assessed_automatic_pseudo_masks_only",
    }
    if predictor is not None:
        import segment_anything

        package = Path(segment_anything.__file__).parent
        report["sam_source_sha256"] = {
            str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))
        }
        if report["sam_source_sha256"] != previous["sam_source_sha256"]:
            raise ValueError("SAM package changed from v1")
    with (args.output_dir / "report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
