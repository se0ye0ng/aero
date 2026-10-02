"""Isolate grid versus farthest-point controls; first replay the saved grid results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
import yaml
from PIL import Image

from aero_ir.registration.local_consensus_warp import fit_image_warp
from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_external_registration_landmarks import summarize, validate_landmarks


def compare_error_arrays(actual, saved):
    expected = np.array([np.inf if v is None else v for v in saved])
    if not np.allclose(actual, expected, atol=1e-8, rtol=0):
        raise ValueError("grid replay differs from saved external errors")


def compare_scalar(actual, expected):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError("grid support replay differs")
    elif not np.isclose(actual, expected, atol=1e-8, rtol=0):
        raise ValueError("grid box proxy replay differs")


def compare_fit(actual, expected):
    for key in ("status", "input_matches", "unique_rgb_matches", "accepted_matches",
                "controls", "control_match_indices"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"grid fit replay differs: {key}")


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/experiment/registration_control_selection_cpu.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = yaml.safe_load(Path(config["external_config"]).read_text())
    anti_config = yaml.safe_load(Path(config["antiuav_config"]).read_text())
    if config["maximum_controls"] != settings["control_grid"]**2:
        raise ValueError("grid and FPS must share the same maximum control budget")
    settings["maximum_controls"] = config["maximum_controls"]
    baselines, hashes = {}, {}
    for name in ("external", "antiuav"):
        path = Path(config[f"{name}_baseline"])
        if file_sha256(path) != config[f"{name}_baseline_sha256"]:
            raise ValueError("unexpected frozen baseline report")
        baselines[name] = read_json(path)
        for key, digest in baselines[name]["input_and_source_sha256"].items():
            if key in hashes and hashes[key] != digest:
                raise ValueError("conflicting baseline provenance")
            hashes[key] = digest
        hashes[str(path)] = file_sha256(path)
    for p in (args.config, Path(__file__),
              Path("src/aero_ir/registration/local_consensus_warp.py"),
              Path(config["external_config"]), Path(config["antiuav_config"])):
        if str(p) in hashes and hashes[str(p)] != file_sha256(p):
            raise ValueError("baseline dependency changed")
        hashes[str(p)] = file_sha256(p)
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError(f"changed input: {name}")
    geometry = yaml.safe_load(Path(settings["geometry_config"]).read_text())
    root = Path(geometry["cache"])
    match_root = Path(geometry["image_only_report"]).parent
    anti_report = read_json(anti_config["source_report"])
    anti_metadata = {r["sequence_id"]: r["inputs"] for r in anti_report["inputs"]}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    external_rows, anti_rows = [], []
    for policy in config["policies"]:
        for old in baselines["external"]["rows"]:
            method, pair = old["method"], old["pair"]
            folder = root/pair
            with Image.open(folder/"V.JPG") as im:
                rgb_size = im.size
            with Image.open(folder/"T.JPG") as im:
                thermal_size = im.size
            with np.load(match_root/f'{method}_{pair.replace("/", "_")}_matches.npz',
                         allow_pickle=False) as data:
                predict, info = fit_image_warp(data["rgb"], data["thermal"], data["confidence"],
                                              rgb_size, thermal_size, settings, policy)
            gt_r = validate_landmarks(np.loadtxt(folder/"points_rgb.txt"), rgb_size[::-1])
            gt_t = validate_landmarks(np.loadtxt(folder/"points_thermal.txt"), thermal_size[::-1])
            if gt_r.shape != gt_t.shape:
                raise ValueError("unpaired reference rows")
            predicted = np.full_like(gt_r, np.nan) if predict is None else predict(gt_r)
            errors = np.linalg.norm(predicted-gt_t, axis=1)
            errors = np.where(np.isfinite(errors), errors, np.inf)
            if policy == "grid":
                compare_fit(info, old["fit"])
                compare_error_arrays(errors, old["errors"])
            external_rows.append(dict(
                policy=policy, method=method, pair=pair, fit=info,
                errors=[float(e) if np.isfinite(e) else None for e in errors],
                summary=summarize(errors, settings["thresholds_thermal_file_pixels"]),
            ))
        for record in anti_report["rows"]:
            if record["condition"] not in anti_config["conditions"]:
                continue
            seq = record["sequence_id"]
            sizes = [tuple(anti_metadata[seq][n]["resized_shape"][::-1])
                     for n in ("visible", "infrared")]
            path = Path(anti_config["source_report"]).parent/record["matches_file"]
            with np.load(path, allow_pickle=False) as data:
                forward, f_info = fit_image_warp(
                    data["points0"], data["points1"], data["confidence"], *sizes, settings, policy)
                reverse, r_info = fit_image_warp(data["reverse0"], data["reverse1"],
                    data["reverse_confidence"], *sizes[::-1], settings, policy)
            boxes = []
            for name in ("visible", "infrared"):
                ann = Path(anti_config["dataset_root"])/"train"/seq/f"{name}.json"
                x, y, w, h = read_json(ann)["gt_rect"][record["frame_index"]]
                entry = anti_metadata[seq][name]
                nh, nw = entry["native_shape"]
                rh, rw = entry["resized_shape"]
                boxes.append(np.array([x, y, x+w, y+h])*[rw/nw, rh/nh, rw/nw, rh/nh])
            values = [corner_iou(forward, *boxes), corner_iou(reverse, *boxes[::-1])]
            if policy == "grid":
                old = next(r for r in baselines["antiuav"]["rows"] if r["sequence_id"] == seq
                           and r["condition"] == record["condition"])
                compare_fit(f_info, old["fit_rgb_to_ir"])
                compare_fit(r_info, old["fit_ir_to_rgb"])
                compare_scalar(values[0], old["corner_envelope_iou_rgb_to_ir"])
                compare_scalar(values[1], old["corner_envelope_iou_ir_to_rgb"])
            anti_rows.append(dict(
                policy=policy, sequence_id=seq, frame_index=record["frame_index"],
                condition=record["condition"], fit_rgb_to_ir=f_info, fit_ir_to_rgb=r_info,
                corner_envelope_iou_rgb_to_ir=values[0], corner_envelope_iou_ir_to_rgb=values[1],
                joint_box_proxy_pass=all(v is not None and v >= anti_config["box_iou_threshold"]
                                         for v in values),
            ))
        print(f"completed external and Anti-UAV policy: {policy}", flush=True)
    external_summary, anti_summary = {}, {}
    for policy in config["policies"]:
        external_summary[policy], anti_summary[policy] = {}, {}
        for method in settings["methods"]:
            selected = [r for r in external_rows if r["policy"] == policy and r["method"] == method]
            external_summary[policy][method] = {
                str(t): float(np.mean([
                    r["summary"]["pck_all_landmarks"][str(t)] for r in selected]))
                for t in settings["thresholds_thermal_file_pixels"]}
        for condition in anti_config["conditions"]:
            selected = [r for r in anti_rows
                        if r["policy"] == policy and r["condition"] == condition]
            if len(selected) != 16:
                raise ValueError("incomplete Anti-UAV transfer panel")
            anti_summary[policy][condition] = dict(
                pairs=len(selected),
                forward_fits=sum(r["fit_rgb_to_ir"]["status"] == "fit" for r in selected),
                both_corner_envelopes=sum(r["corner_envelope_iou_rgb_to_ir"] is not None
                    and r["corner_envelope_iou_ir_to_rgb"] is not None for r in selected),
                joint_box_proxy_passes=sum(r["joint_box_proxy_pass"] for r in selected),
            )
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError("inputs changed during run")
    report = dict(kind=config["kind"], qualification=config["qualification"],
                  grid_replayed_against_saved_results=True, external_rows=external_rows,
                  antiuav_rows=anti_rows, external_summary=external_summary,
                  antiuav_summary=anti_summary, input_and_source_sha256=hashes,
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  limitations=[
                      "Same maximum144 controls, not matched actual control counts.",
                      "Exploratory previously inspected panels, not confirmatory held-out results.",
                      "No changes to consensus, TPS smoothing, hull support or scoring thresholds.",
                      "No topology guarantee, independent Anti-UAV point GT or generator approval.",
                  ])
    with (args.out_dir/"report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(external=external_summary, antiuav=anti_summary), indent=2))


if __name__ == "__main__":
    main()
