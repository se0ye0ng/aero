"""Frozen image-only matchers versus sparse static/ego-motion MS2 references, CPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, via_common_camera
from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.registration.local_consensus_warp import fit_image_warp
from aero_ir.registration.ms2_image_reference import (
    display_thermal,
    score_map,
    sparse_reference,
    thermal_window,
)
from aero_ir.utils.manifest import file_sha256
from scripts.audit_ms2_screen import verified_files
from scripts.probe_external_registration_landmarks import resize_gray, to_native
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_registration_rgb_resolution import ignored_bytecode_paths


def read_image(path):
    with Image.open(path) as image:
        return np.array(image)


def prepare(config):
    hashes = {}
    for name in ("native_report", "motion_report"):
        path = Path(config[name])
        if file_sha256(path) != config[name + "_sha256"]:
            raise ValueError("frozen source report differs")
        report = json.loads(path.read_text())
        for group in ("input_sha256", "source_sha256"):
            for source, digest in report[group].items():
                if file_sha256(source) != digest:
                    raise ValueError(f"changed source prerequisite: {source}")
                hashes[source] = digest
        hashes[str(path)] = file_sha256(path)
    native = json.loads(Path(config["native_report"]).read_text())
    plan_path = Path(config["plan"])
    plan = json.loads(plan_path.read_text())
    if (plan["frame_ids"] != native["frame_ids"] or len(plan["frame_ids"]) != 16
            or native["calibration_numerical_errors"] or native["calibration_required_missing"]
            or native["stereo_depth_dimension_errors"]):
        raise ValueError("incomplete frozen native panel")
    files = {}
    for kind, prefix in (("sync_data", "sync"), ("proj_depth", "depth")):
        files.update(verified_files(Path(config[prefix + "_root"]),
                                    Path(config[prefix + "_report"]), plan_path, kind))
    if {k: file_sha256(v) for k, v in files.items()} != native["extracted_sha256"]:
        raise ValueError("native extracted identities changed")
    hashes.update({str(v): file_sha256(v) for v in files.values()})
    seq = plan["sequence"]
    frames = plan["frame_ids"]
    images = {s: [read_image(files[f"sync_data/{seq}/{s}/img_left/{f}.png"])
                  for f in frames] for s in ("rgb", "thr")}
    window = thermal_window(images["thr"])
    inputs, scales = {}, {}
    for sensor in ("rgb", "thr"):
        gray = ([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images[sensor]] if sensor == "rgb"
                else [display_thermal(im, window) for im in images[sensor]])
        transformed = [resize_gray(im, config["long_side"]) for im in gray]
        inputs[sensor] = [v[0] for v in transformed]
        scales[sensor] = [v[1] for v in transformed]
    c = read_calibration(files[f"sync_data/{seq}/calib.npy"].read_bytes())
    fixed = via_common_camera(from_millimetres(c["R_nir2thr"], c["T_nir2thr"]),
                              from_millimetres(c["R_nir2rgb"], c["T_nir2rgb"]))
    times = {s: parse_timestamps(files[f"sync_data/{seq}/{s}/img_left_timestamp.txt"].read_bytes(),
                                 expected_count=plan["frame_count"]) for s in ("rgb", "thr")}
    poses = read_rgb_poses(Path(config["odom"]), plan)
    references, interpolation = {}, {}
    for frame in frames:
        i = int(frame)
        target_pose, info = interpolate_pose(poses, times["rgb"], int(times["thr"][i]),
                                             max_extrapolation_ns=config["pose_max_extrapolation_ns"])
        interpolation[frame] = info
        transforms = dict(static=fixed, ego=moving_rig_transform(fixed, poses[i], target_pose))
        depths = {s: read_image(files[f"proj_depth/{seq}/{s}/depth/{frame}.png"])
                  for s in ("rgb", "thr")}
        for source, target in (("rgb", "thr"), ("thr", "rgb")):
            references[(frame, source)] = {
                name: sparse_reference(depths[source], depths[target], c[f"K_{source}L"],
                                       c[f"K_{target}L"], transform if source == "rgb"
                                       else np.linalg.inv(transform))
                for name, transform in transforms.items()}
    return plan, images, inputs, scales, window, references, interpolation, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiment/registration_ms2_image_cpu.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if (config["reference_depth_consistency_radius_px"] != 1.
            or config["reference_depth_consistency_abs_m"] != .03
            or config["reference_target_native_pixel_thresholds"] != [1., 3., 5., 10.]
            or config["thermal_input_window"] != "pooled_train16_percentile_0.5_99.5"):
        raise ValueError("unsupported reference/preprocessing configuration")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="aero-ms2-bytecode-")
    sys.dont_write_bytecode = True
    vendor = Path(config["vendor"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != config["vendor_commit"]:
        raise ValueError("vendor revision differs")
    ignored = ignored_bytecode_paths(subprocess.check_output(
        ["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=all"], text=True))
    plan, images, inputs, scales, window, refs, pose_info, hashes = prepare(config)
    settings = yaml.safe_load(Path(config["local_config"]).read_text())
    settings["maximum_controls"] = config["maximum_controls"]
    sources = [args.config, Path(config["local_config"]), Path(__file__),
               Path("scripts/probe_external_registration_landmarks.py"),
               Path("scripts/probe_registration_rgb_resolution.py"),
               Path("src/aero_ir/registration/ms2_image_reference.py"),
               Path("src/aero_ir/registration/detector_free.py"),
               Path("src/aero_ir/registration/local_consensus_warp.py")]
    sources += sorted((vendor / "src").rglob("*.py"))
    hashes.update({str(p): file_sha256(p) for p in sources})
    for spec in config["models"].values():
        if file_sha256(spec["path"]) != spec["sha256"]:
            raise ValueError("model weights differ")
        hashes[spec["path"]] = spec["sha256"]
    reference_files = []
    for (frame, source), variants in refs.items():
        for variant, reference in variants.items():
            path = args.out_dir / f"reference_{frame}_{source}_{variant}.npz"
            np.savez_compressed(path, **reference)
            reference_files.append(dict(path=path.name, sha256=file_sha256(path)))
    # Write preprocessing/identities BEFORE the first inference, not after seeing its scores.
    preflight = dict(config_sha256=file_sha256(args.config), thermal_window_dn=window,
                     frame_ids=plan["frame_ids"], reference_files=reference_files,
                     interpolation=pose_info, input_and_source_sha256=hashes)
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    rows, controls = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for model_name, spec in config["models"].items():
            model = load_matcher("xoftr", Path(spec["path"]), vendor, "cpu")
            gray = inputs["rgb"][0]
            shifted = cv2.warpAffine(gray, np.float32([[1, 0, 8], [0, 1, -8]]), gray.shape[::-1])
            a, b, _, flags = infer(model, "xoftr", gray, shifted, "cpu")
            control_errors = np.linalg.norm(b - a - [8, -8], axis=1)
            controls.append(dict(model=model_name, matches=len(a), flags=flags,
                                 median_input_pixel_error=(float(np.median(control_errors))
                                                           if len(a) else None)))
            for index, frame in enumerate(plan["frame_ids"]):
                other = (index + config["unrelated_frame_cyclic_offset"]) % len(plan["frame_ids"])
                for condition, source, target, target_index in (
                        ("paired", "rgb", "thr", index), ("paired", "thr", "rgb", index),
                        ("unrelated_thermal", "rgb", "thr", other)):
                    a, b, confidence, flags = infer(model, "xoftr", inputs[source][index],
                                                    inputs[target][target_index], "cpu")
                    a = to_native(a, scales[source][index])
                    b = to_native(b, scales[target][target_index])
                    name = f"{model_name}_{frame}_{condition}_{source}"
                    match_path = args.out_dir / f"{name}.npz"
                    np.savez_compressed(match_path, source_xy=a, target_xy=b, confidence=confidence)
                    predict, fit = fit_image_warp(
                        a, b, confidence, images[source][index].shape[:2][::-1],
                        images[target][target_index].shape[:2][::-1], settings,
                        config["control_policy"])
                    scores = {}
                    for variant, reference in refs[(frame, source)].items():
                        predicted = (predict(reference["source_xy"]) if predict is not None else
                                     np.full_like(reference["source_xy"], np.nan))
                        scores[variant] = score_map(predicted, reference)
                    rows.append(dict(model=model_name, frame=frame, condition=condition,
                                     source=source, target=target,
                                     target_frame=plan["frame_ids"][target_index],
                                     flags=flags, fit=fit,
                                     scores=scores, matches=len(a), matches_file=match_path.name,
                                     matches_sha256=file_sha256(match_path)))
                print(f"{model_name} {frame}: paired both directions + unrelated control complete",
                      flush=True)
            del model
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input/source changed during run: {path}")
    result = dict(schema="ms2_image_matching_diagnostic_v1", device="cpu", rows=rows,
                  controls=controls, elapsed_seconds=time.monotonic()-started,
                  preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
                  input_and_source_sha256=hashes, vendor_bytecode_ignored=ignored,
                  note="sparse calibrated references, not independent verified image GT",
                  evaluated_split="train", validation_or_test_access="none",
                  registration_qualified=False, generator_training_approved=False,
                  antiuav_status_changed=False)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"wrote {args.out_dir / 'report.json'}; not registration qualification")


if __name__ == "__main__":
    main()
