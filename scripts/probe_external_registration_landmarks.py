"""Image-only matching scored against external, original-coordinate landmarks.

This small-panel diagnostic never qualifies Anti-UAV or authorizes a generator.
Reference points are not supplied to matching, RANSAC, or model selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.utils.manifest import file_sha256


def resize_gray(gray, long_side):
    h, w = gray.shape
    if min(h, w) < 8 or long_side < 8:
        raise ValueError("image too small")
    scale = min(1.0, long_side / max(h, w))
    new_w, new_h = max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, np.array([new_w / w, new_h / h])


def to_native(points, scale):
    # OpenCV resizing maps pixel centres, not image boundary corners.
    return (np.asarray(points, dtype=np.float64) + 0.5) / scale - 0.5


def validate_landmarks(points, shape):
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 4
            or not np.isfinite(points).all() or (points < 0).any()
            or (points > np.array([shape[1] - 1, shape[0] - 1])).any()):
        raise ValueError("invalid native-coordinate landmarks")
    return points


def project(matrix, points):
    if matrix is None:
        return np.full_like(points, np.nan, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    valid = np.isfinite(homogeneous).all(1) & (np.abs(homogeneous[:, 2]) > 1e-12)
    result = np.full((len(points), 2), np.nan)
    result[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None]
    return result


def estimate(thermal_matches, rgb_matches, config, seed):
    # No reference landmarks or evaluation scores are accepted here.
    if len(thermal_matches) < 4:
        return None, 0
    cv2.setRNGSeed(seed)
    try:
        matrix, mask = cv2.findHomography(
            thermal_matches, rgb_matches, cv2.RANSAC,
            config["ransac_threshold_native_rgb_pixels"],
            maxIters=config["ransac_max_iterations"], confidence=config["ransac_confidence"],
        )
        if (matrix is None or not np.isfinite(matrix).all()
                or np.linalg.matrix_rank(matrix) < 3):
            return None, 0
        return matrix, int(mask.sum())
    except (cv2.error, np.linalg.LinAlgError):
        return None, 0


def landmark_errors(matrix, thermal, rgb):
    rgb_errors = np.linalg.norm(project(matrix, thermal) - rgb, axis=1)
    try:
        inverse = None if matrix is None else np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        inverse = None
    thermal_errors = np.linalg.norm(project(inverse, rgb) - thermal, axis=1)
    return tuple(np.where(np.isfinite(e), e, np.inf) for e in (rgb_errors, thermal_errors))


def summarize(errors, thresholds):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 1 or not len(errors) or (errors < 0).any() or np.isnan(errors).any():
        raise ValueError("nonempty nonnegative errors required; use inf for failures")
    finite = errors[np.isfinite(errors)]
    # Failure cases remain in every PCK denominator. Conditional errors are labelled.
    return {
        "landmarks": len(errors), "unprojectable": int((~np.isfinite(errors)).sum()),
        "conditional_finite_median": float(np.median(finite)) if len(finite) else None,
        "conditional_finite_p95": float(np.percentile(finite, 95)) if len(finite) else None,
        "pck_all_landmarks": {str(t): float(np.mean(errors <= t)) for t in thresholds},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/experiment/registration_external_landmarks_cpu.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["cache"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["config_sha256"] != file_sha256(args.config)
            or manifest["commit"] != config["commit"]):
        raise ValueError("panel configuration changed after download")
    for item in manifest["files"]:
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f"input bytes changed: {item['path']}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    started = time.monotonic()
    rows, pools = [], {}
    provenance = {str(p): file_sha256(p) for p in (
        args.config, Path(__file__), Path("src/aero_ir/registration/detector_free.py"),
        manifest_path,
    )}
    for p in sorted(Path(config["xoftr_vendor"]).rglob("*.py")):
        provenance[str(p)] = file_sha256(p)
    for name in config["methods"]:
        weights = Path(config[f"{name}_weights"])
        provenance[str(weights)] = file_sha256(weights)
        model = load_matcher(name, weights, Path(config["xoftr_vendor"]), args.device)
        for pair_id in manifest["pairs"]:
            folder = root / pair_id
            thermal = cv2.imread(str(folder / "T.JPG"), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(str(folder / "V.JPG"), cv2.IMREAD_GRAYSCALE)
            if thermal is None or rgb is None:
                raise ValueError(f"image decode failed: {pair_id}")
            t_small, t_scale = resize_gray(thermal, config["long_side"])
            r_small, r_scale = resize_gray(rgb, config["long_side"])
            with torch.inference_mode():
                pt, pr, confidence, info = infer(
                    model, name, t_small, r_small, args.device)
            pt, pr = to_native(pt, t_scale), to_native(pr, r_scale)
            safe_id = pair_id.replace("/", "_")
            matches_path = args.out_dir / f"{name}_{safe_id}_matches.npz"
            np.savez_compressed(matches_path, thermal=pt, rgb=pr, confidence=confidence)
            # Only now read reference coordinates: never inputs to inference/estimation.
            gt_t = validate_landmarks(np.loadtxt(folder / "points_thermal.txt"), thermal.shape)
            gt_r = validate_landmarks(np.loadtxt(folder / "points_rgb.txt"), rgb.shape)
            if gt_t.shape != gt_r.shape:
                raise ValueError("unpaired reference landmark rows")
            for seed in config["seeds"]:
                matrix, inliers = estimate(pt, pr, config, seed)
                e_rgb, e_thermal = landmark_errors(matrix, gt_t, gt_r)
                key = (name, seed)
                pools.setdefault(key, []).extend(e_thermal.tolist())
                row = dict(
                    pair=pair_id, method=name, ransac_seed=seed, matches=len(pt),
                    inliers=inliers, match_info=info,
                    thermal_shape=list(thermal.shape), rgb_shape=list(rgb.shape),
                    transform_thermal_to_rgb=None if matrix is None else matrix.tolist(),
                    native_rgb=summarize(e_rgb, [1., 3., 5., 10.]),
                    native_thermal=summarize(e_thermal, config[
                        "report_thresholds_native_thermal_pixels"]),
                    matches_sha256=file_sha256(matches_path),
                )
                rows.append(row)
            median = rows[-1]["native_thermal"]["conditional_finite_median"]
            print(f'{name} {pair_id}: {len(pt)} matches; thermal landmark median {median}',
                  flush=True)
        del model
    for path, expected in provenance.items():
        if file_sha256(path) != expected:
            raise ValueError(f"source/weights changed during run: {path}")
    for item in manifest["files"]:
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise ValueError("input changed during run")
    aggregates = [dict(method=name, ransac_seed=seed,
                       native_thermal=summarize(errors, config[
                           "report_thresholds_native_thermal_pixels"]))
                  for (name, seed), errors in pools.items()]
    for entry in aggregates:
        selected = [r for r in rows if r["method"] == entry["method"]
                    and r["ransac_seed"] == entry["ransac_seed"]]
        entry["pair_macro_pck_native_thermal"] = {
            str(t): float(np.mean([
                r["native_thermal"]["pck_all_landmarks"][str(t)] for r in selected
            ])) for t in config["report_thresholds_native_thermal_pixels"]
        }
    report = dict(kind=config["kind"], status="external_landmarks_measured_not_qualified",
                  qualification=config["qualification"], pairs=manifest["pairs"],
                  device=args.device, torch_version=torch.__version__,
                  opencv_version=cv2.__version__, numpy_version=np.__version__,
                  source_and_weights_sha256=provenance, rows=rows, aggregate=aggregates,
                  elapsed_seconds=time.monotonic()-started,
                  limitations=[
                      "Four fixed scene representatives, not the full dataset or Anti-UAV.",
                      "Human-authored reference landmarks have unquantified annotation error.",
                      "Ground-plane scenes do not establish tiny airborne-target performance.",
                      "Global homography may fail on parallax; failures remain in PCK denominator.",
                      "Seeds repeat RANSAC, not independent trained models or image samples.",
                      "No training, confidence tuning, reference-fitted transform or gate change.",
                      "Data redistribution license not established; do not publish images/points.",
                  ])
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
