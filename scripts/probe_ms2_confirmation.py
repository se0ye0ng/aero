"""Apply already frozen rig corrections on15new training frames, without refitting."""
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

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, project_pixels, via_common_camera
from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.stereo_depth_diagnostic import compute_stereo, sample_disparity
from aero_ir.utils.manifest import file_sha256
from scripts.audit_ms2_screen import verified_files
from scripts.probe_external_registration_landmarks import resize_gray, to_native
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_ms2_stereo_depth import summarize
from scripts.probe_registration_rgb_resolution import ignored_bytecode_paths


def read(path):
    with Image.open(path) as image:
        return np.array(image)


def add_identity(hashes, path):
    """Adding confirmation inputs must never overwrite an earlier frozen identity."""
    key, digest = str(path), file_sha256(path)
    if key in hashes and hashes[key] != digest:
        raise ValueError(f"confirmation dependency differs from frozen candidate: {key}")
    hashes[key] = digest


def candidate_scores(source, target, depth, source_k, target_k, transform, candidates,
                     *, source_shape=(384, 1224), target_shape=(256, 640)):
    """One fixed reference-support denominator for all candidates, including failed projections."""
    reference = project_pixels(source, depth, source_k, target_k, transform,
                               source_shape=source_shape, target_shape=target_shape)
    eligible = reference.supported
    scores = {}
    for name, correction in candidates.items():
        projection = project_pixels(source, depth, source_k, target_k,
                                    np.asarray(correction) @ transform,
                                    source_shape=source_shape, target_shape=target_shape)
        errors = np.linalg.norm(target-projection.target_xy, axis=1)
        errors[~eligible | ~projection.supported] = np.inf
        scores[name] = dict(all_matches=summarize(errors),
                            fixed_supported=summarize(errors[eligible]))
    return scores, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path("configs/experiment/registration_ms2_image_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    plan_path = Path("experiments/ms2_confirmation_plan_01/plan.json")
    if file_sha256(plan_path) != "32c1c923654a46ddfe807905dc4981587d1c1861d8f2a07e81ecdaea9a4f4f1d":
        raise ValueError("frozen confirmation plan changed")
    plan = json.loads(plan_path.read_text())
    source_path = Path("experiments/ms2_stereo_calibration_01/report.json")
    if file_sha256(source_path) != plan["candidate_report_sha256"]:
        raise ValueError("fixed candidate report changed")
    original = json.loads(source_path.read_text())
    expected_transforms = {c["name"]: c["transform"] for c in original["candidates"]
                           if c["name"] in plan["candidate_transforms"]}
    if expected_transforms != plan["candidate_transforms"]:
        raise ValueError("confirmation transforms differ from fitted candidates")
    hashes = original["input_and_source_sha256"].copy()
    extraction_path = Path("experiments/ms2_confirmation_sync_01_report.json")
    extraction = json.loads(extraction_path.read_text())
    if extraction.get("archive_sha256") != (
            "c3a10f2cef0d04ea6999c9885311aa1bda206e3c228960ce3e1462d19e913902"):
        raise ValueError("confirmation extraction comes from a different archive")
    files = verified_files(Path("experiments/ms2_confirmation_sync_01"), extraction_path,
                           plan_path, "sync_data")
    for path in files.values():
        add_identity(hashes, path)
    for path in (config_path, plan_path, source_path, extraction_path, Path(__file__),
                 Path("scripts/probe_ms2_stereo_depth.py")):
        add_identity(hashes, path)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed dependency: {path}")
    base_plan = json.loads(Path(config["plan"]).read_text())
    if (set(plan["frame_ids"]) & set(base_plan["frame_ids"]) or len(plan["frame_ids"]) != 15
            or plan["sequence"] != base_plan["sequence"]):
        raise ValueError("confirmation image IDs overlap or panel differs")
    seq = plan["sequence"]
    calib = read_calibration(files[f"sync_data/{seq}/calib.npy"].read_bytes())
    fixed = via_common_camera(from_millimetres(calib["R_nir2thr"], calib["T_nir2thr"]),
                              from_millimetres(calib["R_nir2rgb"], calib["T_nir2rgb"]))
    fb = calib["K_rgbL"][0, 0] * -calib["T_rgbR"].reshape(3)[0] / 1000.
    if (fb <= 0 or not np.allclose(calib["K_rgbL"], calib["K_rgbR"], atol=1e-9, rtol=0)
            or not np.allclose(calib["R_rgbR"], np.eye(3), atol=1e-9, rtol=0)
            or np.max(np.abs(calib["T_rgbR"].reshape(3)[1:])) > 1e-9):
        raise ValueError("unsupported stereo geometry")
    times = {s: parse_timestamps(files[f"sync_data/{seq}/{s}/img_left_timestamp.txt"].read_bytes(),
                                 expected_count=base_plan["frame_count"]) for s in ("rgb", "thr")}
    rgb_right_times = parse_timestamps(
        files[f"sync_data/{seq}/rgb/img_right_timestamp.txt"].read_bytes(),
        expected_count=base_plan["frame_count"])
    poses = read_rgb_poses(Path(config["odom"]), base_plan)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="aero-ms2-confirm-bytecode-")
    sys.dont_write_bytecode = True
    vendor = Path(config["vendor"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != config["vendor_commit"]:
        raise ValueError("vendor revision changed")
    ignored = ignored_bytecode_paths(subprocess.check_output(
        ["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=all"], text=True))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    window = (3308., 4974.)  # Original window; never refit on confirmation pixels.
    preflight = dict(plan_sha256=file_sha256(plan_path), input_and_source_sha256=hashes,
                     thermal_window_dn=window, wrong_pair_cyclic_offset=7, refit=False,
                     opencv_version=cv2.__version__, torch_version=torch.__version__,
                     qualification=False, comparison_denominator="fixed author-geometry support")
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    prepared, frames = {}, []
    for frame in plan["frame_ids"]:
        rgb, rgb_right = [read(files[f"sync_data/{seq}/rgb/img_{side}/{frame}.png"])
                          for side in ("left", "right")]
        thermal = read(files[f"sync_data/{seq}/thr/img_left/{frame}.png"])
        if (rgb.shape != (384, 1224, 3) or rgb.dtype != np.uint8
                or rgb_right.shape != rgb.shape or rgb_right.dtype != rgb.dtype
                or thermal.shape != (256, 640) or thermal.dtype != np.uint16):
            raise ValueError("unexpected native image dimensions or dtype")
        gray, right_gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (rgb, rgb_right)]
        stereo = compute_stereo(gray, right_gray)
        stereo_path = args.out_dir / f"stereo_{frame}.npz"
        np.savez_compressed(stereo_path, **stereo)
        target_pose, info = interpolate_pose(poses, times["rgb"], int(times["thr"][int(frame)]),
                                             max_extrapolation_ns=20000000)
        transform = moving_rig_transform(fixed, poses[int(frame)], target_pose)
        prepared[frame] = dict(rgb=resize_gray(gray, 640),
                               thr=resize_gray(display_thermal(thermal, window), 640),
                               stereo=stereo, transform=transform)
        frames.append(dict(frame=frame, stereo_file=stereo_path.name,
                           stereo_sha256=file_sha256(stereo_path), pose_info=info,
                           rgb_stereo_skew_ns=int(rgb_right_times[int(frame)]
                                                 - times["rgb"][int(frame)]),
                           thermal_window_clipped_fraction=float(np.mean(
                               (thermal < window[0]) | (thermal > window[1]))),
                           stereo_supported_pixels=int(stereo["valid"].sum()),
                           native_rgb_pixels=gray.size))
    rows = []
    with torch.inference_mode():
        for name, spec in config["models"].items():
            model = load_matcher("xoftr", Path(spec["path"]), vendor, "cpu")
            for index, frame in enumerate(plan["frame_ids"]):
                case = prepared[frame]
                wrong = plan["frame_ids"][(index+7) % 15]
                for condition, target_frame in (("paired", frame), ("unrelated_thermal", wrong)):
                    target = prepared[target_frame]
                    a, b, confidence, flags = infer(model, "xoftr", case["rgb"][0],
                                                    target["thr"][0], "cpu")
                    a, b = to_native(a, case["rgb"][1]), to_native(b, target["thr"][1])
                    disparity, _ = sample_disparity(case["stereo"], a)
                    depth = fb/disparity
                    scores, eligible = candidate_scores(
                        a, b, depth, calib["K_rgbL"], calib["K_thrL"], case["transform"],
                        plan["candidate_transforms"])
                    artifact = args.out_dir / f"{name}_{frame}_{condition}.npz"
                    np.savez_compressed(artifact, source_xy=a, target_xy=b,
                                        confidence=confidence, stereo_depth_m=depth,
                                        author_supported=eligible)
                    rows.append(dict(model=name, frame=frame, condition=condition,
                                     target_frame=target_frame, flags=flags, scores=scores,
                                     artifact=artifact.name, sha256=file_sha256(artifact)))
                print(f"{name} {frame}: fixed candidates + wrong-pair control complete", flush=True)
            del model
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"dependency changed during inference: {path}")
    result = dict(schema="ms2_new_train_frame_confirmation_v1", rows=rows, frames=frames,
                  device="cpu",
                  input_and_source_sha256=hashes, vendor_bytecode_ignored=ignored,
                  preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
                  calibration_refitted=False, registration_qualified=False,
                  generator_training_approved=False, antiuav_status_changed=False,
                  note="new IDs in same training sequence; image-depth estimates, not pixel GT")
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"wrote {args.out_dir / 'report.json'}; not qualification")


if __name__ == "__main__":
    main()
