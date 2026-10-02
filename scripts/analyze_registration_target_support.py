"""Attribute saved Anti-UAV failures without fitting to annotation boxes.

Boxes are used only for post-hoc evidence counts and support attribution. A
match inside both boxes is not independent evidence of correct pixel identity.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import Delaunay

from aero_ir.utils.manifest import file_sha256


def inside(points, box):
    return ((points >= box[:2]).all(1) & (points <= box[2:]).all(1))


def attribute(points, targets, info, source_box, target_box, iou, threshold=.6):
    """Never changes matches, selected controls, fitted predictions or thresholds."""
    points, targets = np.asarray(points), np.asarray(targets)
    source_box, target_box = np.asarray(source_box), np.asarray(target_box)
    source_hits, target_hits = inside(points, source_box), inside(targets, target_box)
    indices = np.asarray(info.get("control_match_indices", []), dtype=int)
    controls = points[indices]
    result = dict(matches=len(points), source_box_matches=int(source_hits.sum()),
                  both_box_matches=int((source_hits & target_hits).sum()),
                  controls_in_source_box=int(inside(controls, source_box).sum()),
                  fit_status=info["status"], corner_iou=iou)
    if info["status"] != "fit":
        result["failure_stage"] = "no_warp_fit"
        return result
    x0, y0, x1, y1 = source_box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    # Translation/isotropic scaling preserve convex-hull membership. Use source
    # coordinates directly; these are the exact controls of the saved fit.
    supported = Delaunay(controls).find_simplex(corners) >= 0
    result["source_corners_inside_control_hull"] = int(supported.sum())
    if not supported.all():
        if iou is not None:
            raise ValueError("saved IoU contradicts unsupported source corners")
        result["failure_stage"] = "source_corner_outside_control_hull"
    elif iou is None:
        result["failure_stage"] = "mapped_corner_unavailable"
    else:
        result["failure_stage"] = "box_proxy_pass" if iou >= threshold else "box_proxy_iou_low"
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = read_json(args.report)
    hashes = dict(report["input_and_source_sha256"])
    for p in (args.report, Path(__file__)):
        hashes[str(p)] = file_sha256(p)
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError(f"changed input: {name}")
    config_path = Path("configs/experiment/registration_antiuav_local_warp_cpu.yaml")
    if str(config_path) not in hashes:
        raise ValueError("unverified source configuration")
    config = yaml.safe_load(config_path.read_text())
    source_path = Path(config["source_report"])
    source = read_json(source_path)
    metadata = {r["sequence_id"]: r["inputs"] for r in source["inputs"]}
    rows = []
    for row in report["antiuav_rows"]:
        seq, frame, condition = row["sequence_id"], row["frame_index"], row["condition"]
        record = next(r for r in source["rows"] if r["sequence_id"] == seq
                      and r["frame_index"] == frame and r["condition"] == condition)
        boxes = []
        for name in ("visible", "infrared"):
            path = Path(config["dataset_root"])/"train"/seq/f"{name}.json"
            if str(path) not in hashes:
                raise ValueError("unverified annotation")
            x, y, w, h = read_json(path)["gt_rect"][frame]
            entry = metadata[seq][name]
            nh, nw = entry["native_shape"]
            rh, rw = entry["resized_shape"]
            boxes.append(np.array([x, y, x+w, y+h])*[rw/nw, rh/nh, rw/nw, rh/nh])
        if not np.allclose(boxes, record["boxes_xyxy"], atol=1e-4, rtol=0):
            raise ValueError("annotation coordinate mismatch")
        path = source_path.parent/record["matches_file"]
        if str(path) not in hashes:
            raise ValueError("unverified matches")
        with np.load(path, allow_pickle=False) as data:
            for direction, keys, ordered_boxes in (
                    ("rgb_to_ir", ("points0", "points1"), boxes),
                    ("ir_to_rgb", ("reverse0", "reverse1"), boxes[::-1])):
                result = attribute(data[keys[0]], data[keys[1]], row[f"fit_{direction}"],
                                   *ordered_boxes, row[f"corner_envelope_iou_{direction}"],
                                   config["box_iou_threshold"])
                rows.append(dict(policy=row["policy"], condition=condition, sequence_id=seq,
                                 frame_index=frame, direction=direction, **result))
    summary = {}
    for policy in ("grid", "fps"):
        summary[policy] = {}
        for condition in config["conditions"]:
            summary[policy][condition] = {}
            for direction in ("rgb_to_ir", "ir_to_rgb"):
                selected = [r for r in rows if r["policy"] == policy
                            and r["condition"] == condition and r["direction"] == direction]
                if len(selected) != 16:
                    raise ValueError("incomplete panel")
                summary[policy][condition][direction] = dict(
                    failure_stages=dict(Counter(r["failure_stage"] for r in selected)),
                    pairs_without_source_box_matches=sum(r["source_box_matches"] == 0
                                                         for r in selected),
                    pairs_without_both_box_matches=sum(r["both_box_matches"] == 0
                                                       for r in selected))
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError(f"input changed during audit: {name}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    result = dict(kind="saved_match_target_support_attribution_v1", rows=rows, summary=summary,
                  input_and_source_sha256=hashes,
                  qualification="posthoc_train_box_proxy_no_physical_or_generator_approval",
                  limitations=[
                      "Annotation boxes used only after fitting, not to select or refit a method.",
                      "Both-box matches are not independently correct pixel correspondences.",
                      "Absent matches in this frozen inference do not prove matching impossible.",
                      "Source hull support does not establish topology or mapped-corner validity.",
                  ])
    with (args.out_dir/"report.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
