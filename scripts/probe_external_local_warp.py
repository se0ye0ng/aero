"""Image-only local-affine consensus plus TPS, evaluated against external references.

All model selection constants are in YAML. No reference coordinates enter control
selection or fitting. Reference query positions only sample the already-fit map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
import yaml
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, QhullError, cKDTree

from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_registration_landmarks import summarize, validate_landmarks


def fit_image_warp(rgb, thermal, confidence, rgb_size, thermal_size, config):
    rgb, thermal, confidence = [np.asarray(x, dtype=np.float64) for x in
                                (rgb, thermal, confidence)]
    if (rgb.ndim != 2 or rgb.shape[1] != 2 or rgb.shape != thermal.shape
            or confidence.shape != (len(rgb),) or not all(
                np.isfinite(x).all() for x in (rgb, thermal, confidence))):
        raise ValueError("invalid image-only match arrays")
    if not 0 < config["trim_fraction"] <= 1:
        raise ValueError("invalid trim fraction")
    # Highest-confidence duplicate wins; reference coordinates play no role.
    order = np.argsort(-confidence, kind="stable")
    _, unique = np.unique(rgb[order], axis=0, return_index=True)
    chosen = order[np.sort(unique)]
    rgb, thermal, confidence = rgb[chosen], thermal[chosen], confidence[chosen]
    info = dict(input_matches=len(order), unique_rgb_matches=len(rgb))
    if len(rgb) < config["minimum_controls"]:
        return None, dict(info, status="insufficient_matches")
    scale_rgb, scale_t = float(max(rgb_size)), float(max(thermal_size))
    x, y = (rgb+.5)/scale_rgb, (thermal+.5)/scale_t
    k = min(config["neighbors"], len(x)-1)
    if k < 4:
        return None, dict(info, status="insufficient_neighbors")
    tree = cKDTree(x)
    neighbors = tree.query(x, k=k+1)[1]
    residual = np.full(len(x), np.inf)
    for i in range(len(x)):
        ids = neighbors[i][neighbors[i] != i][:k]
        design = np.column_stack((x[ids]-x[i], np.ones(len(ids))))
        target = y[ids]
        selected = np.arange(len(ids))
        # Point i is excluded: a false match cannot validate itself by interpolation.
        for _ in range(config["trim_iterations"]):
            coef, _, rank, _ = np.linalg.lstsq(design[selected], target[selected], rcond=None)
            if rank < 3:
                break
            errors = np.linalg.norm(design@coef-target, axis=1)
            count = max(4, int(np.ceil(config["trim_fraction"]*len(ids))))
            selected = np.argsort(errors, kind="stable")[:count]
        else:
            coef, _, rank, _ = np.linalg.lstsq(design[selected], target[selected], rcond=None)
            if rank == 3:
                residual[i] = np.linalg.norm(coef[-1]-y[i])*scale_t
    threshold = max(config["agreement_min_thermal_pixels"],
                    config["agreement_thermal_long_side_fraction"]*scale_t)
    accepted = np.flatnonzero(residual <= threshold)
    # Uniform image-grid coverage; top confidence per cell, not best reference error.
    cell = np.floor((rgb+.5)/np.asarray(rgb_size)*config["control_grid"]).astype(int)
    controls, occupied = [], set()
    for i in accepted:  # still in descending confidence order
        key = tuple(cell[i])
        if key not in occupied:
            controls.append(i)
            occupied.add(key)
    info.update(accepted_matches=len(accepted), controls=len(controls),
                agreement_threshold_thermal_pixels=threshold,
                control_match_indices=chosen[controls].tolist())
    if len(controls) < config["minimum_controls"]:
        return None, dict(info, status="insufficient_controls")
    try:
        fit_x, fit_y = x[controls], y[controls]
        interpolator = RBFInterpolator(fit_x, fit_y, kernel="thin_plate_spline", degree=1,
                                      smoothing=config["tps_smoothing"])
        hull = Delaunay(fit_x)
    except (ValueError, np.linalg.LinAlgError, QhullError):
        return None, dict(info, status="degenerate_controls")

    def predict(query):
        q = (np.asarray(query, dtype=np.float64)+.5)/scale_rgb
        out = np.full_like(q, np.nan)
        supported = hull.find_simplex(q) >= 0
        out[supported] = interpolator(q[supported])*scale_t-.5
        # Unsupported extrapolation and out-of-frame outputs count as failures.
        supported &= np.isfinite(out).all(1) & (out >= 0).all(1)
        supported &= (out <= np.asarray(thermal_size)-1).all(1)
        out[~supported] = np.nan
        return out

    return predict, dict(info, status="fit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/experiment/registration_external_local_warp_cpu.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    geometry_path = Path(config["geometry_config"])
    previous = yaml.safe_load(geometry_path.read_text())
    root = Path(previous["cache"])
    manifest_path = root/"manifest.json"
    report_path = Path(previous["image_only_report"])
    if (file_sha256(manifest_path) != previous["manifest_sha256"]
            or file_sha256(report_path) != previous["image_only_report_sha256"]):
        raise ValueError("frozen upstream evidence changed")
    manifest, prior = (json.loads(p.read_text()) for p in (manifest_path, report_path))
    inputs = {str(root/f["path"]): f["sha256"] for f in manifest["files"]}
    inputs.update({str(p): file_sha256(p) for p in (
        args.config, geometry_path, manifest_path, report_path, Path(__file__),
        Path("scripts/probe_external_registration_landmarks.py"),
    )})
    for r in prior["rows"]:
        name = f'{r["method"]}_{r["pair"].replace("/", "_")}_matches.npz'
        inputs[str(report_path.parent/name)] = r["matches_sha256"]
    for path, expected in inputs.items():
        if file_sha256(path) != expected:
            raise ValueError(f"changed input: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for method in config["methods"]:
        for pair in manifest["pairs"]:
            folder = root/pair
            with Image.open(folder/"V.JPG") as image:
                rgb_size = image.size
            with Image.open(folder/"T.JPG") as image:
                thermal_size = image.size
            matches = report_path.parent/f'{method}_{pair.replace("/", "_")}_matches.npz'
            with np.load(matches, allow_pickle=False) as data:
                predict, info = fit_image_warp(data["rgb"], data["thermal"], data["confidence"],
                                              rgb_size, thermal_size, config)
            # Reference points are loaded only after control selection and fitting.
            rgb = validate_landmarks(np.loadtxt(folder/"points_rgb.txt"), rgb_size[::-1])
            thermal = validate_landmarks(np.loadtxt(folder/"points_thermal.txt"),
                                         thermal_size[::-1])
            if rgb.shape != thermal.shape:
                raise ValueError("unpaired reference points")
            predicted = np.full_like(rgb, np.nan) if predict is None else predict(rgb)
            errors = np.linalg.norm(predicted-thermal, axis=1)
            errors = np.where(np.isfinite(errors), errors, np.inf)
            rows.append(dict(method=method, pair=pair, fit=info,
                             errors=[float(e) if np.isfinite(e) else None for e in errors],
                             summary=summarize(errors, config["thresholds_thermal_file_pixels"])))
            print(method, pair, info["status"], rows[-1]["summary"], flush=True)
    aggregate = {method: {
        str(t): float(np.mean([r["summary"]["pck_all_landmarks"][str(t)]
                              for r in rows if r["method"] == method]))
        for t in config["thresholds_thermal_file_pixels"]} for method in config["methods"]}
    for path, expected in inputs.items():
        if file_sha256(path) != expected:
            raise ValueError("inputs changed during run")
    report = dict(kind=config["kind"], qualification=config["qualification"], rows=rows,
                  pair_macro_pck=aggregate, input_and_source_sha256=inputs,
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  limitations=[
                      "Exploratory four-pair panel already seen during earlier analysis.",
                      "Joint filtering/coverage/TPS change, not an isolated architecture ablation.",
                      "A fitted TPS need not be invertible; no topology/cycle qualification.",
                      "Outside control hull and out-of-frame predictions count as failures.",
                      "No Anti-UAV inference, training, or generator authorization.",
                  ])
    with (args.out_dir/"report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
