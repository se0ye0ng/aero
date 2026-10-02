"""CPU-only non-deterioration ablation on immutable fresh SAM v2 mask artifacts."""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.silhouette_pareto import compare
from aero_ir.utils.manifest import file_sha256


def load_inputs(path):
    report = json.loads(path.read_text())
    if (
        report.get("experiment") != "sam_centred_roi_similarity_v2"
        or report.get("mode") != "extract"
        or report.get("evaluated_split") != "train"
        or report.get("validation_or_test_access") != "none"
    ):
        raise ValueError("requires fresh train-only SAM v2 extraction, not cached replay")
    root = path.parent.resolve()
    for name, digest in report["artifacts_sha256"].items():
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root) or file_sha256(artifact) != digest:
            raise ValueError(f"invalid source artifact: {name}")
    data = {}
    for row in report["mask_rows"]:
        name = row["artifact"]
        if name not in report["artifacts_sha256"]:
            raise ValueError("mask artifact has no recorded hash")
        key = (row["sequence_id"], row["modality"])
        if key in data:
            raise ValueError("duplicate mask row")
        with np.load(root / name, allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ("masks", "excluded")}
        data[key]["meta"] = row["crop_metadata"]
    ids = [r["sequence_id"] for r in report["inputs"]]
    if (
        len(ids) < 2
        or len(set(ids)) != len(ids)
        or set(data) != {(i, m) for i in ids for m in ("rgb", "ir")}
    ):
        raise ValueError("incomplete mask panel")
    expected = {
        (kind, sid, ids[j])
        for i, sid in enumerate(ids)
        for kind, j in (("paired", i), ("shuffled", (i + 1) % len(ids)))
    }
    keys = [(r["kind"], r["rgb_sequence"], r["ir_sequence"]) for r in report["rows"]]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("incomplete or changed paired/shuffled panel")
    return report, data


def summarize(rows):
    summary = {}
    for method in ("unconstrained", "non_deteriorating"):
        summary[method] = {}
        for kind in ("paired", "shuffled"):
            group = [r for r in rows if r["kind"] == kind]
            available = [r for r in group if "score" in r["result"].get(method, {})]
            common = [r for r in available if r["result"]["baseline"] is not None]
            perturbations = [
                p
                for r in available
                for p in r["result"][method]["perturbed_prompt_checks_not_gt"]
                if p["baseline"] is not None and p["selected"] is not None
            ]
            stats = {
                "panel_size": len(group),
                "available": len(available),
                "identity_fallback": sum(
                    r["result"][method]["status"] == "identity_fallback" for r in available
                ),
                "comparable_count": len(common),
                "perturbed_comparable": len(perturbations),
                "perturbed_chamfer_worsened": sum(
                    p["selected"]["chamfer_model_px"] > p["baseline"]["chamfer_model_px"] + 1e-10
                    for p in perturbations
                ),
            }
            for metric in ("dice_loss", "chamfer_model_px"):
                stats[metric] = {
                    "median_before": float(
                        np.median([r["result"]["baseline"][metric] for r in common])
                    )
                    if common
                    else None,
                    "median_after": float(
                        np.median([r["result"][method]["score"][metric] for r in common])
                    )
                    if common
                    else None,
                    "worsened": sum(
                        r["result"][method]["score"][metric]
                        > r["result"]["baseline"][metric] + 1e-10
                        for r in common
                    ),
                }
            summary[method][kind] = stats
        pairs = {
            kind: {
                r["rgb_sequence"]: r["result"][method]["score"]
                for r in rows
                if r["kind"] == kind and "score" in r["result"].get(method, {})
            }
            for kind in ("paired", "shuffled")
        }
        common = sorted(pairs["paired"].keys() & pairs["shuffled"].keys())
        summary[method]["common_sources"] = {
            "count": len(common),
            "paired_lower_dice": sum(
                pairs["paired"][k]["dice_loss"] < pairs["shuffled"][k]["dice_loss"] - 1e-10
                for k in common
            ),
            "paired_lower_chamfer": sum(
                pairs["paired"][k]["chamfer_model_px"]
                < pairs["shuffled"][k]["chamfer_model_px"] - 1e-10
                for k in common
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("experiments/registration_sam_train16_v2/report.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refusing to overwrite existing experiment")
    source, data = load_inputs(args.source_report)
    source_digest = file_sha256(args.source_report)
    cv2.setNumThreads(1)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    files = [
        "scripts/probe_registration_sam_pareto.py",
        "src/aero_ir/registration/silhouette_pareto.py",
        "src/aero_ir/registration/silhouette_search.py",
        "src/aero_ir/registration/roi_geometry.py",
    ]
    artifacts = {}
    source_hashes = {s: file_sha256(project / s) for s in files}
    for name in files:
        target = args.output_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
        artifacts[str(target.relative_to(args.output_dir))] = file_sha256(target)
    rows = []
    for i, row in enumerate(source["rows"]):
        old = row["result"]
        result = (
            compare(data[row["rgb_sequence"], "rgb"], data[row["ir_sequence"], "ir"], old)
            if old["status"] == "fit_pseudo_masks_only"
            else {"status": old["status"]}
        )
        rows.append(
            {**{k: row[k] for k in ("kind", "rgb_sequence", "ir_sequence")}, "result": result}
        )
        print(f"{i + 1}/{len(source['rows'])} {row['kind']} {row['rgb_sequence']}", flush=True)
    if file_sha256(args.source_report) != source_digest or any(
        file_sha256(project / s) != h for s, h in source_hashes.items()
    ):
        raise ValueError("input report or source changed during execution")
    report = {
        "experiment": "sam_v2_matched_candidates_non_deterioration_v1",
        "source_report": str(args.source_report.resolve()),
        "source_report_sha256": source_digest,
        "mask_artifacts_sha256": {
            k: v for k, v in source["artifacts_sha256"].items() if k.endswith(".npz")
        },
        "sources": source_hashes,
        "artifacts_sha256": artifacts,
        "protocol": {
            "candidate_pool": "identity + saved v2 optimum + 3^4 neighbours",
            "scale_factor": 1.05,
            "angle_step_degrees": 3,
            "translation_step_pixels": 2,
            "same_candidates_for_both_methods": True,
            "constraints": "neither original-prompt Dice nor Chamfer may worsen from identity",
            "unsafe_baseline": "abstain, not a successful fit",
            "prompt_perturbations_used_for_selection": False,
            "original_eligibility_preserved": True,
            "development_panel_already_observed": True,
        },
        "summary": summarize(rows),
        "rows": rows,
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "qualification": "not_assessed_automatic_pseudo_masks_only",
        "elapsed_seconds": time.monotonic() - started,
    }
    with (args.output_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
