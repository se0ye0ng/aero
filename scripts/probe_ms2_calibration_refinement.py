"""Exploratory fixed-rig calibration candidates; never overwrite author calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, ms2_depth_metres, via_common_camera
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_ego_motion import read_rgb_poses

ROTATION_SCALE = np.deg2rad(3.)
TRANSLATION_SCALE = .1


def correction(parameters):
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape not in ((3,), (6,)) or not np.isfinite(parameters).all():
        raise ValueError("finite3/6 normalized parameters required")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(parameters[:3]*ROTATION_SCALE).as_matrix()
    if len(parameters) == 6:
        transform[:3, 3] = parameters[3:]*TRANSLATION_SCALE
    return transform


def predict(points, intrinsics, transform):
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(xyz).all() or (xyz[:, 2] <= 0).any():
        raise ValueError("candidate places a measured point behind target camera")
    homogeneous = xyz @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def fit(points, observed, intrinsics, dimensions):
    if len(points) < 12:
        raise ValueError("insufficient source-depth-supported matches for exploratory fit")

    def residual(p):
        return (predict(points, intrinsics, correction(p)) - observed).ravel()

    result = least_squares(residual, np.zeros(dimensions), bounds=(-1., 1.),
                           loss="soft_l1", f_scale=1., max_nfev=1000,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    return dict(transform=correction(result.x).tolist(), parameters=result.x.tolist(),
                rotation_vector_deg=np.rad2deg(result.x[:3]*ROTATION_SCALE).tolist(),
                translation_m=correction(result.x)[:3, 3].tolist(),
                success=bool(result.success), message=result.message, nfev=result.nfev,
                active_bounds=result.active_mask.tolist(),
                normalized_jacobian_singular=singular.tolist(),
                normalized_jacobian_rank=int(np.linalg.matrix_rank(result.jac)),
                matches=len(points), cost=float(result.cost))


def score(points, observed, intrinsics, transform):
    errors = np.linalg.norm(predict(points, intrinsics, transform) - observed, axis=1)
    return dict(matches=len(points), median_px=float(np.median(errors)) if len(errors) else None,
                within_px={str(t): float(np.mean(errors <= t)) if len(errors) else None
                           for t in (1., 3., 5., 10.)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/experiment/registration_ms2_image_cpu.yaml").read_text())
    report_path = Path("experiments/ms2_image_matching_01/report.json")
    expected = "3abbeca131b191c85c24357ea6758c0acae9dcbdd51fffd5a066e3718d717d39"
    if file_sha256(report_path) != expected:
        raise ValueError("original screen changed")
    report = json.loads(report_path.read_text())
    hashes = report["input_and_source_sha256"].copy()
    hashes[str(report_path)] = expected
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
    panels = dict(fit=plan["frame_ids"][::2], check=plan["frame_ids"][1::2])
    cases = []
    for row in report["rows"]:
        if row["source"] != "rgb" or row["condition"] != "paired":
            continue
        frame = row["frame"]
        match_path = report_path.parent / row["matches_file"]
        if (match_path.parent != report_path.parent
                or file_sha256(match_path) != row["matches_sha256"]):
            raise ValueError("unsafe/modified image matches")
        hashes[str(match_path)] = row["matches_sha256"]
        with np.load(match_path, allow_pickle=False) as arrays:
            a, b = arrays["source_xy"], arrays["target_xy"]
        depth_path = (Path(config["depth_root"]) / "proj_depth" / plan["sequence"]
                      / "rgb/depth" / f"{frame}.png")
        with Image.open(depth_path) as image:
            depth = ms2_depth_metres(np.array(image))
        y, x = np.nonzero(np.isfinite(depth))
        distance, nearest = cKDTree(np.c_[x, y]).query(a)
        associated = distance <= .5
        points = (np.c_[a[associated], np.ones(np.count_nonzero(associated))]
                  @ np.linalg.inv(calib["K_rgbL"]).T)
        points *= depth[y[nearest[associated]], x[nearest[associated]], None]
        pose, info = interpolate_pose(poses, times["rgb"], int(times["thr"][int(frame)]),
                                      max_extrapolation_ns=config["pose_max_extrapolation_ns"])
        transform = moving_rig_transform(fixed, poses[int(frame)], pose)
        current_thermal = points @ transform[:3, :3].T + transform[:3, 3]
        cases.append(dict(model=row["model"], frame=frame,
                          split="fit" if frame in panels["fit"] else "check",
                          points=current_thermal, observed=b[associated],
                          source_matches=len(a), pose_info=info))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(input_and_source_sha256=hashes, panels=panels,
                     candidate_models=["rotation_only", "rigid6"], source_depth_radius_px=.5,
                     bounds_per_axis=dict(rotation_degrees=3., translation_m=.1),
                     loss="soft_l1", loss_scale_target_px=1., fit_weight="equal per match",
                     note="post-screen exploratory train-frame split; NOT unseen test data",
                     qualification=False, unchanged_intrinsics=True, unchanged_timestamps=True)
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    candidates = [dict(name="author_calibration", transform=np.eye(4).tolist())]
    for model in config["models"]:
        training = [x for x in cases if x["model"] == model and x["split"] == "fit"]
        points = np.concatenate([x["points"] for x in training])
        observed = np.concatenate([x["observed"] for x in training])
        for kind, dimensions in (("rotation_only", 3), ("rigid6", 6)):
            candidate = fit(points, observed, calib["K_thrL"], dimensions)
            candidate.update(name=f"{model}_{kind}", fit_matcher=model)
            candidates.append(candidate)
    rows = []
    for candidate in candidates:
        for case in cases:
            metrics = score(case["points"], case["observed"], calib["K_thrL"],
                            np.asarray(candidate["transform"]))
            rows.append({k: case[k] for k in ("model", "frame", "split", "source_matches")}
                        | dict(candidate=candidate["name"], **metrics))
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"dependency changed: {path}")
    result = dict(schema="ms2_exploratory_rigid_calibration_v1", rows=rows, candidates=candidates,
                  input_and_source_sha256=hashes,
                  preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
                  registration_qualified=False, generator_training_approved=False,
                  note="image/depth-supported candidate fit, not independent physical pixel GT",
                  native_calibration_overwritten=False)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
