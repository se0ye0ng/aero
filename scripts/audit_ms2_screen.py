"""Inspect the frozen MS2 train panel's native arrays, calibration and timestamps.

Requires completed, hashed selective extractions. This reports observations and
format/coordinate prerequisites, not physical registration qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_depth_consistency import depth_product_consistency
from aero_ir.data.ms2_timestamps import compare_timestamps, parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, intrinsic_matrix, via_common_camera
from aero_ir.utils.manifest import file_sha256
from scripts.audit_ms2_metadata import audit


def verified_files(root: Path, report_path: Path, plan_path: Path, kind: str) -> dict[str, Path]:
    report = json.loads(report_path.read_text())
    plan = json.loads(plan_path.read_text())
    if (report.get("schema") != "ms2_selected_extraction_v1"
            or report.get("complete") is not True or report.get("required_missing") != []
            or report.get("kind") != kind or report.get("sequence") != plan["sequence"]
            or report.get("plan_sha256") != file_sha256(plan_path)):
        raise ValueError("incomplete extraction or different frozen plan")
    result = {}
    for item in report["files"]:
        name = item["path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or path.parts[:2] != (kind, plan["sequence"]) or name in result):
            raise ValueError("invalid or duplicate extraction path")
        local = root / name
        if not local.resolve().is_relative_to(root.resolve()):
            raise ValueError("extraction path escaped root")
        if local.stat().st_size != item["bytes"] or file_sha256(local) != item["sha256"]:
            raise ValueError(f"extracted input changed: {name}")
        result[name] = local
    return result


def inspect(sync_root: Path, depth_root: Path, sync_report: Path, depth_report: Path,
            plan_path: Path, metadata: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    source = audit(metadata)
    if (plan.get("schema") != "ms2_train_screen_plan_v1"
            or plan["sequence"] != source["first_training_sequence"]):
        raise ValueError("expected predetermined first official training sequence")
    files = {**verified_files(sync_root, sync_report, plan_path, "sync_data"),
             **verified_files(depth_root, depth_report, plan_path, "proj_depth")}
    sequence = plan["sequence"]
    calibration = read_calibration(files[f"sync_data/{sequence}/calib.npy"].read_bytes())
    checks = {}
    for key, value in calibration.items():
        try:
            if key.startswith("K_"):
                intrinsic_matrix(value)
            elif key.startswith("R_"):
                from_millimetres(value, calibration[f"T_{key[2:]}"])
        except (KeyError, ValueError) as error:
            checks[key] = str(error)
    required = {"K_rgbL", "K_thrL", "R_nir2rgb", "T_nir2rgb", "R_nir2thr", "T_nir2thr"}
    missing = sorted(required - calibration.keys())
    images, dimension_errors, depth_checks = [], [], []
    transforms = {}
    if not checks and not missing:
        rgb = from_millimetres(calibration["R_nir2rgb"], calibration["T_nir2rgb"])
        thr = from_millimetres(calibration["R_nir2thr"], calibration["T_nir2thr"])
        transforms = {("rgb", "thr"): via_common_camera(thr, rgb),
                      ("thr", "rgb"): via_common_camera(rgb, thr)}
    for frame in plan["frame_ids"]:
        depths = {}
        for sensor in ("rgb", "thr"):
            shapes = {}
            for folder, kind in (("img_left", "sync_data"), ("img_right", "sync_data"),
                                 ("depth", "proj_depth")):
                name = f"{kind}/{sequence}/{sensor}/{folder}/{frame}.png"
                with Image.open(files[name]) as image:
                    image.load()
                    array = np.array(image)
                is_depth = folder == "depth"
                expected = np.dtype("uint16" if is_depth or sensor == "thr" else "uint8")
                if array.dtype != expected or array.ndim != (3 if sensor == "rgb" and
                                                            not is_depth else 2):
                    raise ValueError(f"unexpected native image dtype or dimensions: {name}")
                if array.ndim == 3 and array.shape[2] != 3:
                    raise ValueError(f"expected three RGB channels: {name}")
                shapes[folder] = array.shape[:2]
                item = dict(path=name, shape=list(array.shape), dtype=str(array.dtype),
                            minimum=int(array.min()), maximum=int(array.max()))
                if is_depth:
                    depths[sensor] = array
                    valid = array[array > 0].astype(np.float64) / 256.
                    quantiles = (np.percentile(valid, [0, 5, 50, 95, 100]).tolist()
                                 if len(valid) else [])
                    item.update(valid_pixels=len(valid), valid_fraction=len(valid) / array.size,
                                depth_m_percentiles=quantiles)
                images.append(item)
            if len(set(shapes.values())) != 1:
                dimension_errors.append(dict(frame=frame, sensor=sensor,
                                             shapes={k: list(v) for k, v in shapes.items()}))
        for (source_sensor, target_sensor), transform in transforms.items():
            depth_checks.append(dict(
                frame=frame, source=source_sensor, target=target_sensor,
                **depth_product_consistency(depths[source_sensor], depths[target_sensor],
                                            calibration[f"K_{source_sensor}L"],
                                            calibration[f"K_{target_sensor}L"], transform)))
    timestamps = {}
    for sensor in ("rgb", "thr"):
        name = f"sync_data/{sequence}/{sensor}/img_left_timestamp.txt"
        timestamps[sensor] = parse_timestamps(files[name].read_bytes(),
                                              expected_count=plan["frame_count"])
    indices = [int(frame) for frame in plan["frame_ids"]]
    additional_times, missing_times = {}, []
    timestamp_paths = {
        "rgb_right": "rgb/img_right_timestamp.txt",
        "thr_right": "thr/img_right_timestamp.txt",
        "lidar_left": "lidar/left_timestamp.txt",
        "lidar_right": "lidar/right_timestamp.txt",
        "gps_imu": "gps_imu/data_timestamp.txt",
    }
    for name, relative in timestamp_paths.items():
        key = f"sync_data/{sequence}/{relative}"
        if key not in files:
            missing_times.append(relative)
            continue
        values = parse_timestamps(files[key].read_bytes(), expected_count=plan["frame_count"])
        additional_times[name] = compare_timestamps(timestamps["rgb"], values)
    return dict(
        schema="ms2_native_panel_inventory_v1", sequence=sequence,
        frame_ids=plan["frame_ids"], images=images,
        stereo_depth_dimension_errors=dimension_errors,
        depth_product_consistency=depth_checks,
        depth_consistency_skipped_invalid_calibration=not bool(transforms),
        calibration={k: v.tolist() for k, v in calibration.items()},
        calibration_numerical_errors=checks, calibration_required_missing=missing,
        extrinsic_translation_source_unit="mm", calibration_convention_verified=False,
        same_index_timestamps=compare_timestamps(timestamps["rgb"], timestamps["thr"]),
        selected_timestamps=compare_timestamps(timestamps["rgb"][indices],
                                               timestamps["thr"][indices]),
        selected_thermal_minus_rgb_ns={frame: int(timestamps["thr"][int(frame)] -
                                                 timestamps["rgb"][int(frame)])
                                       for frame in plan["frame_ids"]},
        additional_sensor_times_relative_to_rgb_left=additional_times,
        additional_timestamp_files_missing=missing_times,
        input_sha256={str(p): file_sha256(p) for p in (plan_path, sync_report, depth_report)},
        extracted_sha256={k: file_sha256(v) for k, v in files.items()},
        registration_qualified=False, generator_training_approved=False,
        antiuav_status_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sync-root", "depth-root", "sync-report", "depth-report", "plan", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=Path("experiments/external/ms2_metadata"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = inspect(args.sync_root, args.depth_root, args.sync_report, args.depth_report,
                     args.plan, args.metadata)
    root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        name: file_sha256(root / name) for name in (
            "scripts/audit_ms2_screen.py", "src/aero_ir/data/ms2_calibration.py",
            "src/aero_ir/data/ms2_depth_consistency.py",
            "src/aero_ir/data/ms2_timestamps.py", "src/aero_ir/registration/calibrated.py")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out}; native inventory only, not registration approval")


if __name__ == "__main__":
    main()
