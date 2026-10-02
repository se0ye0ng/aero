"""Transfer frozen external local-warp settings to saved Anti-UAV train matches.

Corner-envelope IoU is an engineering proxy, not dense registration qualification.
No images are re-inferred and no annotation enters image-warp fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_local_warp import fit_image_warp


def corner_iou(predict, source_box, target_box):
    if predict is None:
        return None
    x0, y0, x1, y1 = source_box
    warped = predict(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    if not np.isfinite(warped).all():
        return None
    lo, hi = warped.min(0), warped.max(0)
    target_box = np.asarray(target_box)
    overlap = np.maximum(0, np.minimum(hi, target_box[2:])-np.maximum(lo, target_box[:2])).prod()
    union = (hi-lo).prod()+(target_box[2:]-target_box[:2]).prod()-overlap
    return float(overlap/union) if union > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/experiment/registration_antiuav_local_warp_cpu.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings_path = Path(config["local_warp_config"])
    settings = yaml.safe_load(settings_path.read_text())
    source_path = Path(config["source_report"])
    if file_sha256(source_path) != config["source_report_sha256"]:
        raise ValueError("unexpected saved match panel")
    source = json.loads(source_path.read_text())
    if source["evaluated_split"] != "train" or source["validation_or_test_access"] != "none":
        raise ValueError("only frozen train matches are supported")
    provenance = {r["sequence_id"]: r["inputs"] for r in source["inputs"]}
    hashes = {str(p): file_sha256(p) for p in (
        args.config, settings_path, source_path, Path(__file__),
        Path("scripts/probe_external_local_warp.py"),
    )}
    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for record in source["rows"]:
        if record["condition"] not in config["conditions"]:
            continue
        seq = record["sequence_id"]
        path = source_path.parent/record["matches_file"]
        hashes[str(path)] = record["matches_sha256"]
        if file_sha256(path) != hashes[str(path)]:
            raise ValueError("saved matches changed")
        sizes = [tuple(provenance[seq][name]["resized_shape"][::-1])
                 for name in ("visible", "infrared")]
        with np.load(path, allow_pickle=False) as data:
            forward, f_info = fit_image_warp(data["points0"], data["points1"], data["confidence"],
                                            *sizes, settings)
            reverse, r_info = fit_image_warp(data["reverse0"], data["reverse1"],
                                            data["reverse_confidence"], *sizes[::-1], settings)
        for info in (f_info, r_info):
            if "agreement_threshold_thermal_pixels" in info:
                info["agreement_threshold_target_grid_pixels"] = info.pop(
                    "agreement_threshold_thermal_pixels")
        # Only after fitting, read and verify the annotations used for proxy scoring.
        boxes = []
        for name in ("visible", "infrared"):
            annotation = Path(config["dataset_root"])/"train"/seq/f"{name}.json"
            info = provenance[seq][name]
            hashes[str(annotation)] = info["annotation_sha256"]
            if file_sha256(annotation) != hashes[str(annotation)]:
                raise ValueError("annotation provenance changed")
            x, y, w, h = json.loads(annotation.read_text())["gt_rect"][record["frame_index"]]
            nh, nw = info["native_shape"]
            rh, rw = info["resized_shape"]
            box = np.array([x, y, x+w, y+h])*[rw/nw, rh/nh, rw/nw, rh/nh]
            boxes.append(box)
        if not np.allclose(boxes, record["boxes_xyxy"], atol=1e-4, rtol=0):
            raise ValueError("stored box coordinate contract differs")
        values = [corner_iou(forward, *boxes), corner_iou(reverse, *boxes[::-1])]
        rows.append(dict(sequence_id=seq, frame_index=record["frame_index"],
                         condition=record["condition"], target_fully_retained=record[
                             "target_fully_retained"], fit_rgb_to_ir=f_info, fit_ir_to_rgb=r_info,
                         corner_envelope_iou_rgb_to_ir=values[0],
                         corner_envelope_iou_ir_to_rgb=values[1],
                         joint_box_proxy_pass=all(v is not None and v >= config[
                             "box_iou_threshold"] for v in values)))
    summary = {}
    for condition in config["conditions"]:
        selected = [r for r in rows if r["condition"] == condition]
        if len(selected) != 16 or len({r["sequence_id"] for r in selected}) != 16:
            raise ValueError("incomplete fixed16 transfer panel")
        summary[condition] = dict(
            pairs=len(selected),
            joint_box_proxy_passes=sum(r["joint_box_proxy_pass"] for r in selected),
            forward_fit_available=sum(r["fit_rgb_to_ir"]["status"] == "fit" for r in selected),
            both_corner_envelopes_available=sum(
                r["corner_envelope_iou_rgb_to_ir"] is not None
                and r["corner_envelope_iou_ir_to_rgb"] is not None for r in selected),
        )
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError("input changed during diagnostic")
    result = dict(kind=config["kind"], qualification=config["qualification"],
                  rows=rows, summary=summary, input_and_source_sha256=hashes,
                  limitations=[
                      "Train16 transfer diagnostic, not independent physical accuracy.",
                      "Uses stored resized match coordinates, not new native-image inference.",
                      "External candidate settings unchanged; no target-box tuning.",
                      "Reverse agreement thresholds use the target RGB working grid.",
                      "Directional TPS maps need not be inverses or topology-safe.",
                      "Corner envelopes approximate nonlinear transfer; no dense gate computed.",
                      "Unsupported corners count as failures, not excluded samples.",
                  ])
    with (args.out_dir/"report.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
