"""Compare independent RGB-only stereo depth with MS2 projected LiDAR depth, CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.spatial import cKDTree

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import (
    from_millimetres,
    ms2_depth_metres,
    project_pixels,
    via_common_camera,
)
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.registration.stereo_depth_diagnostic import (
    SGBM_SETTINGS,
    compute_stereo,
    sample_disparity,
)
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_ego_motion import read_rgb_poses


def read(path):
    with Image.open(path) as image:
        return np.array(image)


def summarize(errors):
    errors = np.asarray(errors)
    finite = errors[np.isfinite(errors)]
    return dict(points=len(errors), supported=len(finite),
                conditional_median_px=float(np.median(finite)) if len(finite) else None,
                fraction_all={str(t): float(np.mean(errors <= t)) if len(errors) else None
                              for t in (1., 3., 5., 10.)},
                fraction_supported={str(t): float(np.mean(finite <= t)) if len(finite) else None
                                    for t in (1., 3., 5., 10.)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/experiment/registration_ms2_image_cpu.yaml").read_text())
    screen_path = Path("experiments/ms2_image_matching_01/report.json")
    expected = "3abbeca131b191c85c24357ea6758c0acae9dcbdd51fffd5a066e3718d717d39"
    if file_sha256(screen_path) != expected:
        raise ValueError("original image screen changed")
    screen = json.loads(screen_path.read_text())
    hashes = screen["input_and_source_sha256"].copy()
    for path in (screen_path, Path(__file__),
                 Path("src/aero_ir/registration/stereo_depth_diagnostic.py")):
        hashes[str(path)] = file_sha256(path)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed dependency: {path}")
    plan = json.loads(Path(config["plan"]).read_text())
    root = Path(config["sync_root"]) / "sync_data" / plan["sequence"]
    depth_root = Path(config["depth_root"]) / "proj_depth" / plan["sequence"]
    calib = read_calibration((root / "calib.npy").read_bytes())
    if (not np.allclose(calib["K_rgbL"], calib["K_rgbR"], atol=1e-9, rtol=0)
            or not np.allclose(calib["R_rgbR"], np.eye(3), atol=1e-9, rtol=0)
            or np.max(np.abs(calib["T_rgbR"].reshape(3)[1:])) > 1e-9):
        raise ValueError("rectified equal-intrinsic horizontal stereo required")
    stereo_transform = from_millimetres(calib["R_rgbR"], calib["T_rgbR"])
    baseline = -stereo_transform[0, 3]
    if baseline <= 0:
        raise ValueError("positive left-minus-right disparity convention required")
    fb = calib["K_rgbL"][0, 0]*baseline
    fixed = via_common_camera(from_millimetres(calib["R_nir2thr"], calib["T_nir2thr"]),
                              from_millimetres(calib["R_nir2rgb"], calib["T_nir2rgb"]))
    times = {s: parse_timestamps((root / s / "img_left_timestamp.txt").read_bytes(),
                                 expected_count=plan["frame_count"]) for s in ("rgb", "thr")}
    right_times = parse_timestamps((root / "rgb/img_right_timestamp.txt").read_bytes(),
                                   expected_count=plan["frame_count"])
    poses = read_rgb_poses(Path(config["odom"]), plan)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(input_and_source_sha256=hashes, frames=plan["frame_ids"],
                     opencv_version=cv2.__version__, settings=SGBM_SETTINGS,
                     input_resolution="native",
                     lr_tolerance_px=1., max_four_neighbor_disparity_spread_px=1.,
                     lidar_source_radius_px=.5, positive_control_disparity_px=12.,
                     depth_role="RGB-only image estimate, NOT measured ground truth",
                     qualification=False)
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    rows, frames, controls = [], [], []
    for index, frame in enumerate(plan["frame_ids"]):
        left, right = [cv2.cvtColor(read(root / "rgb" / f"img_{side}" / f"{frame}.png"),
                                    cv2.COLOR_RGB2GRAY) for side in ("left", "right")]
        if index == 0:
            synthetic_right = np.zeros_like(left)
            synthetic_right[:, :-12] = left[:, 12:]
            control = compute_stereo(left, synthetic_right)
            control_errors = np.abs(control["disparity"]-12.)
            control_errors[~control["valid"]] = np.inf
            controls.append(summarize(control_errors.ravel()))
        stereo = compute_stereo(left, right)
        stereo_path = args.out_dir / f"stereo_{frame}.npz"
        np.savez_compressed(stereo_path, **stereo)
        depth = ms2_depth_metres(read(depth_root / "rgb/depth" / f"{frame}.png"))
        y, x = np.nonzero(np.isfinite(depth))
        native_xy = np.c_[x, y].astype(np.float64)
        native_disparity, native_supported = sample_disparity(stereo, native_xy)
        lidar_disparity = fb/depth[y, x]
        disparity_error = np.abs(native_disparity-lidar_disparity)
        disparity_error[~native_supported] = np.inf
        target_pose, info = interpolate_pose(
            poses, times["rgb"], int(times["thr"][int(frame)]),
            max_extrapolation_ns=config["pose_max_extrapolation_ns"])
        transform = moving_rig_transform(fixed, poses[int(frame)], target_pose)
        # Quantify the small RGB left/right timing effect rather than assuming perfect synchrony.
        right_pose, right_info = interpolate_pose(
            poses, times["rgb"], int(right_times[int(frame)]),
            max_extrapolation_ns=config["pose_max_extrapolation_ns"])
        timed_stereo = moving_rig_transform(stereo_transform, poses[int(frame)], right_pose)
        static_projection, timed_projection = [project_pixels(
            native_xy, depth[y, x], calib["K_rgbL"], calib["K_rgbR"], t,
            source_shape=left.shape, target_shape=right.shape) for t in
            (stereo_transform, timed_stereo)]
        timing_distance = np.linalg.norm(static_projection.target_xy-timed_projection.target_xy,
                                         axis=1)
        frames.append(dict(frame=frame, native_pixels=left.size,
                           stereo_supported_pixels=int(stereo["valid"].sum()),
                           stereo_file=stereo_path.name, sha256=file_sha256(stereo_path),
                           lidar_vs_stereo_disparity=summarize(disparity_error),
                           rgb_stereo_skew_ns=int(right_times[int(frame)]-times["rgb"][int(frame)]),
                           rgb_stereo_motion_displacement_max_px=float(np.max(timing_distance)),
                           rgb_stereo_motion_displacement_median_px=float(np.median(timing_distance)),
                           rgb_right_pose=right_info, thermal_pose=info))
        tree = cKDTree(native_xy)
        selected = [row for row in screen["rows"]
                    if row["frame"] == frame and row["source"] == "rgb"]
        for row in selected:
            path = screen_path.parent / row["matches_file"]
            if path.parent != screen_path.parent or file_sha256(path) != row["matches_sha256"]:
                raise ValueError("unsafe or modified image match artifact")
            hashes[str(path)] = row["matches_sha256"]
            with np.load(path, allow_pickle=False) as data:
                a, b = data["source_xy"], data["target_xy"]
            distance, nearest = tree.query(a)
            lidar_available = distance <= .5
            disparity, stereo_available = sample_disparity(stereo, a)
            stereo_depth = fb/disparity
            lidar_depth = depth[y[nearest], x[nearest]].copy()
            lidar_depth[~lidar_available] = np.nan
            results, projections, error_arrays = {}, {}, {}
            for kind, z in (("stereo", stereo_depth), ("lidar", lidar_depth)):
                projection = project_pixels(a, z, calib["K_rgbL"], calib["K_thrL"], transform,
                                            source_shape=left.shape, target_shape=(256, 640))
                errors = np.linalg.norm(b-projection.target_xy, axis=1)
                errors[~projection.supported] = np.inf
                projections[kind], error_arrays[kind] = projection.target_xy, errors
                results[kind] = summarize(errors)
            common = np.isfinite(error_arrays["stereo"]) & np.isfinite(error_arrays["lidar"])
            results["common_support"] = {kind: summarize(errors[common])
                                         for kind, errors in error_arrays.items()}
            artifact = args.out_dir / f"{row['model']}_{frame}_{row['condition']}.npz"
            np.savez_compressed(artifact, source_xy=a, target_xy=b,
                                stereo_target_xy=projections["stereo"],
                                lidar_target_xy=projections["lidar"],
                                stereo_error=error_arrays["stereo"],
                                lidar_error=error_arrays["lidar"], common_support=common,
                                stereo_depth_m=stereo_depth, lidar_depth_m=lidar_depth)
            rows.append(dict(model=row["model"], frame=frame, condition=row["condition"],
                             matches=len(a), stereo_available=int(stereo_available.sum()),
                             lidar_available=int(lidar_available.sum()), scores=results,
                             artifact=artifact.name, sha256=file_sha256(artifact)))
        print(f"{frame}: stereo depth + same-point depth-source comparison complete", flush=True)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"dependency changed during run: {path}")
    result = dict(schema="ms2_rgb_stereo_depth_diagnostic_v1", frames=frames, rows=rows,
                  controls=controls, input_and_source_sha256=hashes, opencv_version=cv2.__version__,
                  preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
                  registration_qualified=False, generator_training_approved=False,
                  note="RGB stereo estimate independent of IR matcher; not physical pixel GT")
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
