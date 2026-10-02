"""Same-modality stereo controls separating image/calibration/depth disagreement."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
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
from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_epipolar import fundamental_matrix, line_distances, summary
from scripts.probe_external_registration_landmarks import resize_gray, to_native
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_registration_rgb_resolution import ignored_bytecode_paths


def read(path):
    with Image.open(path) as image:
        return np.array(image)


def camera_pose(rgb_pose, sensor_from_rgb):
    """World-from-sensor given world-from-RGB, all at one declared timestamp."""
    return rgb_pose @ np.linalg.inv(sensor_from_rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path("configs/experiment/registration_ms2_image_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    screen_path = Path("experiments/ms2_image_matching_01/report.json")
    expected = "3abbeca131b191c85c24357ea6758c0acae9dcbdd51fffd5a066e3718d717d39"
    if file_sha256(screen_path) != expected:
        raise ValueError("original image report changed")
    screen = json.loads(screen_path.read_text())
    hashes = screen["input_and_source_sha256"].copy()
    preflight_path = screen_path.parent / "preflight.json"
    if file_sha256(preflight_path) != screen["preflight_sha256"]:
        raise ValueError("original preflight changed")
    window = json.loads(preflight_path.read_text())["thermal_window_dn"]
    for path in (screen_path, preflight_path, Path(__file__),
                 Path("scripts/analyze_ms2_epipolar.py")):
        hashes[str(path)] = file_sha256(path)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed dependency: {path}")
    plan = json.loads(Path(config["plan"]).read_text())
    root = Path(config["sync_root"]) / "sync_data" / plan["sequence"]
    depth_root = Path(config["depth_root"]) / "proj_depth" / plan["sequence"]
    calib = read_calibration((root / "calib.npy").read_bytes())
    poses = read_rgb_poses(Path(config["odom"]), plan)
    times = {(s, side): parse_timestamps((root / s / f"img_{side}_timestamp.txt").read_bytes(),
                                         expected_count=plan["frame_count"])
             for s in ("rgb", "thr") for side in ("left", "right")}
    sensor_from_rgb = dict(rgb=np.eye(4), thr=via_common_camera(
        from_millimetres(calib["R_nir2thr"], calib["T_nir2thr"]),
        from_millimetres(calib["R_nir2rgb"], calib["T_nir2rgb"])))
    cases = []
    for sensor in ("rgb", "thr"):
        stereo = from_millimetres(calib[f"R_{sensor}R"], calib[f"T_{sensor}R"])
        for frame in plan["frame_ids"]:
            pose_info, sensor_poses = {}, {}
            for side in ("left", "right"):
                try:
                    pose, info = interpolate_pose(
                        poses, times[("rgb", "left")], int(times[(sensor, side)][int(frame)]),
                        max_extrapolation_ns=config["pose_max_extrapolation_ns"])
                    sensor_poses[side] = camera_pose(pose, sensor_from_rgb[sensor])
                    pose_info[side] = info
                except ValueError as exc:
                    pose_info[side] = dict(unavailable=str(exc))
            transforms = dict(static=stereo)
            if len(sensor_poses) == 2:
                transforms["ego"] = moving_rig_transform(
                    stereo, sensor_poses["left"], sensor_poses["right"])
            skew = int(times[(sensor, "right")][int(frame)]
                       - times[(sensor, "left")][int(frame)])
            cases.append(dict(sensor=sensor, frame=frame, transforms=transforms,
                              pose_info=pose_info, stereo_skew_ns=skew))
    vendor = Path(config["vendor"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != config["vendor_commit"]:
        raise ValueError("changed vendor revision")
    ignored = ignored_bytecode_paths(subprocess.check_output(
        ["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=all"], text=True))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="aero-ms2-stereo-bytecode-")
    sys.dont_write_bytecode = True
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(input_and_source_sha256=hashes, frames=plan["frame_ids"],
                     thermal_window_dn=window, pose_max_extrapolation_ns=20000000,
                     note="post-screen stereo diagnostic, no threshold selection or qualification")
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    rows = []
    with torch.inference_mode():
        for name, spec in config["models"].items():
            model = load_matcher("xoftr", Path(spec["path"]), vendor, "cpu")
            for case in cases:
                sensor, frame = case["sensor"], case["frame"]
                native = [read(root / sensor / f"img_{side}" / f"{frame}.png")
                          for side in ("left", "right")]
                gray = [(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if sensor == "rgb"
                         else display_thermal(im, window)) for im in native]
                inputs = [resize_gray(g, config["long_side"]) for g in gray]
                a, b, confidence, flags = infer(model, "xoftr", inputs[0][0], inputs[1][0], "cpu")
                a, b = to_native(a, inputs[0][1]), to_native(b, inputs[1][1])
                artifact = args.out_dir / f"{name}_{sensor}_{frame}.npz"
                np.savez_compressed(artifact, source_xy=a, target_xy=b, confidence=confidence)
                depth = ms2_depth_metres(read(depth_root / sensor / "depth" / f"{frame}.png"))
                y, x = np.nonzero(np.isfinite(depth))
                distance, index = cKDTree(np.c_[x, y]).query(a)
                scores = {}
                for variant, transform in case["transforms"].items():
                    k0, k1 = calib[f"K_{sensor}L"], calib[f"K_{sensor}R"]
                    d0, d1 = line_distances(a, b, fundamental_matrix(k0, k1, transform))
                    # Nearest depth approximation, not an exact subpixel surface measurement.
                    associated = distance <= .5
                    projection = project_pixels(
                        a[associated], depth[y[index[associated]], x[index[associated]]],
                        k0, k1, transform, source_shape=gray[0].shape, target_shape=gray[1].shape)
                    errors = np.linalg.norm(b[associated] - projection.target_xy, axis=1)
                    errors[~projection.supported] = np.inf
                    scores[variant] = dict(source_line=summary(d0), target_line=summary(d1),
                                           depth_projection_nearest_0_5_source_px=summary(errors))
                rows.append(dict(model=name, sensor=sensor, frame=frame, flags=flags,
                                 stereo_skew_ns=case["stereo_skew_ns"],
                                 pose_info=case["pose_info"], scores=scores,
                                 artifact=artifact.name, sha256=file_sha256(artifact)))
                print(f"{name} {sensor} {frame}: stereo complete", flush=True)
            del model
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"dependency changed during run: {path}")
    result = dict(schema="ms2_same_modality_stereo_diagnostic_v1", rows=rows, device="cpu",
                  input_and_source_sha256=hashes, vendor_bytecode_ignored=ignored,
                  preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
                  registration_qualified=False, generator_training_approved=False)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
