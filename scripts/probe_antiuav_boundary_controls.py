"""Fixed boundary augmentation transfer; train box proxy only, never qualification."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import QhullError

from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_external_boundary_controls import accepted_matches, augment_boundary
from scripts.probe_external_local_warp import fit_image_warp
from scripts.probe_external_orientation_repair import selected_tps


def augmented_fit(x, y, confidence, sizes, settings):
    baseline, info = fit_image_warp(x, y, confidence, *sizes, settings)
    result = dict(original_fit=info, status=info["status"])
    if baseline is None:
        return baseline, None, result
    accepted, grid = accepted_matches(x, y, confidence, *sizes, settings)
    np.testing.assert_array_equal(grid, info["control_match_indices"])
    if len(accepted) != info["accepted_matches"]:
        raise ValueError("consensus replay differs")
    ids = augment_boundary(x, accepted, grid)
    result.update(
        controls_before=len(grid), controls_after=len(ids), expanded_match_indices=ids.tolist()
    )
    try:
        warp = selected_tps(x[ids], y[ids], *sizes, settings["tps_smoothing"])
    except (QhullError, ValueError, np.linalg.LinAlgError):
        result["status"] = "expanded_fit_failed"
        warp = None
    return baseline, warp, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config_path = Path("configs/experiment/registration_antiuav_local_warp_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    settings_path, source_path = Path(config["local_warp_config"]), Path(config["source_report"])
    if file_sha256(source_path) != config["source_report_sha256"]:
        raise ValueError("source changed")
    settings, source = (
        yaml.safe_load(settings_path.read_text()),
        json.loads(source_path.read_text()),
    )
    if source["evaluated_split"] != "train" or source["validation_or_test_access"] != "none":
        raise ValueError("train only")
    provenance = {r["sequence_id"]: r["inputs"] for r in source["inputs"]}
    files = [config_path, settings_path, source_path, Path(__file__)] + [
        Path("scripts") / name
        for name in (
            "probe_antiuav_local_warp.py",
            "probe_external_boundary_controls.py",
            "probe_external_local_warp.py",
            "probe_external_orientation_repair.py",
        )
    ]
    hashes = {str(p): file_sha256(p) for p in files}
    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for record in source["rows"]:
        if record["condition"] not in config["conditions"]:
            continue
        seq = record["sequence_id"]
        path = source_path.parent / record["matches_file"]
        if (
            path.resolve().parent != source_path.parent.resolve()
            or file_sha256(path) != record["matches_sha256"]
        ):
            raise ValueError("unsafe or changed matches")
        hashes[str(path)] = record["matches_sha256"]
        sizes = [tuple(provenance[seq][n]["resized_shape"][::-1]) for n in ("visible", "infrared")]
        with np.load(path, allow_pickle=False) as a:
            f = augmented_fit(a["points0"], a["points1"], a["confidence"], sizes, settings)
            r = augmented_fit(
                a["reverse0"], a["reverse1"], a["reverse_confidence"], sizes[::-1], settings
            )
        boxes = []
        for name in ("visible", "infrared"):
            annotation = Path(config["dataset_root"]) / "train" / seq / f"{name}.json"
            info = provenance[seq][name]
            hashes[str(annotation)] = info["annotation_sha256"]
            if file_sha256(annotation) != info["annotation_sha256"]:
                raise ValueError("annotations changed")
            x, y, w, h = json.loads(annotation.read_text())["gt_rect"][record["frame_index"]]
            nh, nw = info["native_shape"]
            rh, rw = info["resized_shape"]
            boxes.append(np.array([x, y, x + w, y + h]) * [rw / nw, rh / nh, rw / nw, rh / nh])
        np.testing.assert_allclose(boxes, record["boxes_xyxy"], atol=1e-4, rtol=0)
        scores = {}
        for index, name in enumerate(("baseline", "boundary")):
            values = [corner_iou(f[index], *boxes), corner_iou(r[index], *boxes[::-1])]
            scores[name] = dict(
                iou=values,
                joint_pass=all(v is not None and v >= config["box_iou_threshold"] for v in values),
            )
        rows.append(
            dict(
                sequence_id=seq,
                frame_index=record["frame_index"],
                condition=record["condition"],
                forward=f[2],
                reverse=r[2],
                scores=scores,
            )
        )
    summary = {}
    for condition in config["conditions"]:
        selected = [r for r in rows if r["condition"] == condition]
        if len(selected) != 16 or len({r["sequence_id"] for r in selected}) != 16:
            raise ValueError("incomplete panel")
        summary[condition] = {
            name: dict(
                pairs=16,
                joint_passes=sum(r["scores"][name]["joint_pass"] for r in selected),
                both_available=sum(
                    all(v is not None for v in r["scores"][name]["iou"]) for r in selected
                ),
            )
            for name in ("baseline", "boundary")
        }
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed input: {p}")
    result = dict(
        rows=rows,
        summary=summary,
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Train16 box proxy, not independent pixel accuracy.",
            "All unsupported corners count as failures.",
            "No target-box-guided control fitting or parameter tuning.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
