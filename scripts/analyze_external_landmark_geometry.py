"""Cross-fit external reference geometry, never an image-only registration method.

Each reference is predicted once from other references per seed. Fits consume GT
and are deliberately labelled oracle diagnostics, not deployed model performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import scipy
import yaml
from PIL import Image
from scipy.interpolate import RBFInterpolator

from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_registration_landmarks import project, summarize, validate_landmarks


def folds(n, count, seed):
    if type(count) is not int or not 2 <= count <= n or n - (n + count - 1)//count < 4:
        raise ValueError("at least four fitting points and nonempty held-out folds required")
    order = np.random.default_rng(seed).permutation(n)
    for test in np.array_split(order, count):
        train = np.setdiff1d(np.arange(n), test)
        yield train, test


def fit_predict(family, source, target, query, source_size, target_size, smoothing):
    """Fit RGB→thermal directly; query target coordinates are never accepted."""
    source, target, query = (np.asarray(v, dtype=np.float64) for v in (source, target, query))
    if (source.ndim != 2 or source.shape[1] != 2 or source.shape != target.shape
            or len(source) < 4 or query.ndim != 2 or query.shape[1] != 2
            or not all(np.isfinite(v).all() for v in (source, target, query))):
        raise ValueError("invalid fitting/query coordinates")
    if min(source_size) < 1 or min(target_size) < 1:
        raise ValueError("positive image dimensions required")
    a, b = float(max(source_size)), float(max(target_size))
    x, y, q = (source+.5)/a, (target+.5)/b, (query+.5)/a
    if np.linalg.matrix_rank(np.column_stack((x, np.ones(len(x))))) < 3:
        raise ValueError("collinear fitting points")
    if family == "affine":
        coef, _, rank, _ = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)
        if rank < 3:
            raise ValueError("rank-deficient affine fit")
        prediction = np.column_stack((q, np.ones(len(q)))) @ coef
    elif family == "homography":
        matrix, _ = cv2.findHomography(x, y, method=0)
        if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
            raise ValueError("invalid reference homography")
        prediction = project(matrix, q)
    elif family == "thin_plate_spline":
        if len(np.unique(x, axis=0)) != len(x):
            raise ValueError("duplicate TPS fitting coordinates")
        prediction = RBFInterpolator(x, y, kernel="thin_plate_spline", degree=1,
                                     smoothing=smoothing)(q)
    else:
        raise ValueError("unknown family")
    return prediction*b-.5


def cross_fit(source, target, source_size, target_size, family, count, seed, smoothing):
    prediction = np.full_like(target, np.nan, dtype=np.float64)
    coverage = np.zeros(len(target), dtype=int)
    records = []
    for train, test in folds(len(source), count, seed):
        failure = None
        try:
            predicted = fit_predict(family, source[train], target[train], source[test],
                                    source_size, target_size, smoothing)
            prediction[test] = predicted
        except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
            failure = str(exc)
        coverage[test] += 1
        records.append(dict(train=train.tolist(), test=test.tolist(), failure=failure))
    if not np.all(coverage == 1):
        raise ValueError("cross-fit coverage is not exactly once per landmark")
    errors = np.linalg.norm(prediction-target, axis=1)
    return np.where(np.isfinite(errors), errors, np.inf), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/experiment/registration_external_geometry_cpu.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["cache"])
    manifest_path = root / "manifest.json"
    image_report_path = Path(config["image_only_report"])
    if file_sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("unexpected reference manifest")
    if file_sha256(image_report_path) != config["image_only_report_sha256"]:
        raise ValueError("unexpected image-only report")
    manifest = json.loads(manifest_path.read_text())
    image_report = json.loads(image_report_path.read_text())
    if manifest["pairs"] != image_report["pairs"]:
        raise ValueError("reference and image-only panels differ")
    inputs = {str(root/f["path"]): f["sha256"] for f in manifest["files"]}
    inputs.update({str(p): file_sha256(p) for p in (
        args.config, Path(__file__), Path("scripts/probe_external_registration_landmarks.py"),
        manifest_path, image_report_path,
    )})
    for path, expected in inputs.items():
        if file_sha256(path) != expected:
            raise ValueError(f"changed input: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    rows = []
    for pair in manifest["pairs"]:
        folder = root / pair
        with Image.open(folder / "V.JPG") as image:
            rgb_size = image.size
        with Image.open(folder / "T.JPG") as image:
            thermal_size = image.size
        rgb = validate_landmarks(np.loadtxt(folder / "points_rgb.txt"), rgb_size[::-1])
        thermal = validate_landmarks(np.loadtxt(folder / "points_thermal.txt"), thermal_size[::-1])
        if rgb.shape != thermal.shape:
            raise ValueError("unpaired landmark rows")
        for family in config["families"]:
            for seed in config["seeds"]:
                errors, partitions = cross_fit(
                    rgb, thermal, rgb_size, thermal_size, family, config["folds"],
                    seed, config["tps_smoothing"],
                )
                rows.append(dict(
                    pair=pair, family=family, partition_seed=seed, folds=partitions,
                    errors_thermal_file_pixels=[
                        float(e) if np.isfinite(e) else None for e in errors],
                    summary=summarize(errors, config["thresholds_thermal_file_pixels"]),
                ))
        print(f"cross-fit complete: {pair}", flush=True)
    aggregates = []
    for family in config["families"]:
        for seed in config["seeds"]:
            selected = [r for r in rows if r["family"] == family and r["partition_seed"] == seed]
            aggregates.append(dict(
                family=family, partition_seed=seed,
                pair_macro_pck={str(t): float(np.mean([
                    r["summary"]["pck_all_landmarks"][str(t)] for r in selected
                ])) for t in config["thresholds_thermal_file_pixels"]},
            ))
    for path, expected in inputs.items():
        if file_sha256(path) != expected:
            raise ValueError(f"input changed during analysis: {path}")
    report = dict(
        kind=config["kind"], status="reference_cross_fit_complete_not_an_automatic_method",
        qualification=config["qualification"], rows=rows, aggregate=aggregates,
        input_and_source_sha256=inputs,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        opencv_version=cv2.__version__,
        limitations=[
            "Fits use external reference coordinates; they are not image-only predictions.",
            "Point-level cross-validation within four scenes, not held-out-scene generalization.",
            "Direct RGB-to-thermal fitting differs from inverting the image-fitted homography.",
            "TPS is not constrained invertible and does not pass cycle/topology qualification.",
            "Reference uncertainty and nonplanarity are not separately identifiable here.",
            "No Anti-UAV gate, original trained weights, or generator authorization changed.",
        ],
    )
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
