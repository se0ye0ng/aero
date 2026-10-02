"""Depth-free calibrated epipolar residuals of saved MS2 image matches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, via_common_camera
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_ego_motion import read_rgb_poses


def fundamental_matrix(source_k, target_k, transform):
    t = transform[:3, 3]
    cross = np.array([[0., -t[2], t[1]], [t[2], 0., -t[0]], [-t[1], t[0], 0.]])
    return np.linalg.inv(target_k).T @ cross @ transform[:3, :3] @ np.linalg.inv(source_k)


def line_distances(source, target, matrix):
    """Both point-to-line distances, each in its own native pixel units."""
    a, b = np.c_[source, np.ones(len(source))], np.c_[target, np.ones(len(target))]
    target_lines, source_lines = a @ matrix.T, b @ matrix
    numerator = np.abs(np.sum(b * target_lines, axis=1))
    result = []
    for lines in (source_lines, target_lines):
        norm = np.linalg.norm(lines[:, :2], axis=1)
        distance = np.full(len(source), np.inf)
        valid = np.isfinite(norm) & (norm > 1e-12) & np.isfinite(numerator)
        distance[valid] = numerator[valid] / norm[valid]
        result.append(distance)
    return tuple(result)


def summary(errors):
    finite = errors[np.isfinite(errors)]
    return dict(matches=len(errors), invalid=int(np.count_nonzero(~np.isfinite(errors))),
                conditional_median_px=float(np.median(finite)) if len(finite) else None,
                within_px={str(t): float(np.mean(errors <= t)) if len(errors) else None
                           for t in (1., 3., 5., 10.)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config_path = Path("configs/experiment/registration_ms2_image_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    report_path = Path("experiments/ms2_image_matching_01/report.json")
    expected = "3abbeca131b191c85c24357ea6758c0acae9dcbdd51fffd5a066e3718d717d39"
    if file_sha256(report_path) != expected:
        raise ValueError("original image report changed")
    report = json.loads(report_path.read_text())
    hashes = report["input_and_source_sha256"].copy()
    hashes[str(report_path)] = file_sha256(report_path)
    hashes[str(Path(__file__))] = file_sha256(__file__)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed dependency: {path}")
    plan = json.loads(Path(config["plan"]).read_text())
    root = Path(config["sync_root"]) / "sync_data" / plan["sequence"]
    calib = read_calibration((root / "calib.npy").read_bytes())
    fixed = via_common_camera(from_millimetres(calib["R_nir2thr"], calib["T_nir2thr"]),
                              from_millimetres(calib["R_nir2rgb"], calib["T_nir2rgb"]))
    times = {s: parse_timestamps((root / s / "img_left_timestamp.txt").read_bytes(),
                                 expected_count=plan["frame_count"]) for s in ("rgb", "thr")}
    poses = read_rgb_poses(Path(config["odom"]), plan)
    transforms = {}
    for frame in plan["frame_ids"]:
        i = int(frame)
        pose, _ = interpolate_pose(poses, times["rgb"], int(times["thr"][i]),
                                   max_extrapolation_ns=config["pose_max_extrapolation_ns"])
        transforms[frame] = dict(static=fixed, ego=moving_rig_transform(fixed, poses[i], pose))
    rows = []
    for row in report["rows"]:
        path = report_path.parent / row["matches_file"]
        if path.parent != report_path.parent or file_sha256(path) != row["matches_sha256"]:
            raise ValueError("changed or unsafe match path")
        hashes[str(path)] = row["matches_sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            a, b = arrays["source_xy"], arrays["target_xy"]
        source, target = row["source"], row["target"]
        scores = {}
        for variant, transform in transforms[row["frame"]].items():
            if source == "thr":
                transform = np.linalg.inv(transform)
            matrix = fundamental_matrix(calib[f"K_{source}L"], calib[f"K_{target}L"], transform)
            d0, d1 = line_distances(a, b, matrix)
            scores[variant] = dict(source=summary(d0), target=summary(d1))
        rows.append({k: row[k] for k in ("model", "frame", "condition", "source", "target")}
                    | dict(scores=scores))
    result = dict(schema="ms2_depth_free_epipolar_v1", rows=rows,
                  input_and_source_sha256=hashes,
                  note="post-screen diagnostic; line proximity cannot certify position along line",
                  unrelated_control="wrong image; correct-pair geometry used intentionally",
                  registration_qualified=False, generator_training_approved=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
