"""Compare static and timestamp-aware geometry on the frozen MS2 training panel.

Exploratory convention test. Nearest depth pixels are not known same-return
correspondences, and internal depth consistency is not image qualification.
"""
from __future__ import annotations

import argparse
import bz2
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from aero_ir.data.ms2_archive import checked_members, pose_errors
from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import (
    from_millimetres,
    ms2_depth_metres,
    project_pixels,
    via_common_camera,
)
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.utils.manifest import file_sha256
from scripts.audit_ms2_screen import verified_files

DEPTH_BINS_M = (.01, .03, .1, .3)  # Exploratory reporting bins, NOT qualification thresholds.


def read_rgb_poses(path: Path, plan: dict) -> np.ndarray:
    before = file_sha256(path)
    if before != plan["odometry_archive_sha256"]:
        raise ValueError("odometry archive differs from frozen plan")
    poses = {}
    with bz2.open(path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member, name in checked_members(archive, plan["sequence"]):
                if member.isfile() and len(name.parts) == 3 and name.parts[1] == "rgb":
                    data = archive.extractfile(member).read()
                    pose_errors(data)  # finite shape/row check; no orthogonalization
                    values = np.fromstring(data.decode("ascii"), sep=" ")
                    poses[int(name.stem)] = values.reshape(4, 4)
        while stream.read(1024 * 1024):
            pass
    if set(poses) != set(range(plan["frame_count"])) or file_sha256(path) != before:
        raise ValueError("odometry IDs or archive identity changed")
    return np.stack([poses[i] for i in range(plan["frame_count"])])


def measure(source, target, source_k, target_k, transform):
    source_depth, target_depth = ms2_depth_metres(source), ms2_depth_metres(target)
    sy, sx = np.nonzero(np.isfinite(source_depth))
    ty, tx = np.nonzero(np.isfinite(target_depth))
    xy = np.column_stack((sx, sy))
    projection = project_pixels(xy, source_depth[sy, sx], source_k, target_k, transform,
                                source_shape=source.shape, target_shape=target.shape)
    selected = np.flatnonzero(projection.supported)
    associated = np.zeros(len(xy), dtype=bool)
    errors = np.full(len(xy), np.nan)
    if len(tx) and len(selected):
        tree = cKDTree(np.column_stack((tx, ty)))
        distances, nearest = tree.query(projection.target_xy[selected])
        good = distances <= 1.
        indices = selected[good]
        associated[indices] = True
        errors[indices] = np.abs(projection.target_z_m[indices] -
                                 target_depth[ty[nearest[good]], tx[nearest[good]]])
    finite = errors[associated]
    result = dict(source_points=len(xy), projected_supported=len(selected),
                  associated_points=int(associated.sum()),
                  conditional_abs_depth_m_p50_p90=(np.percentile(finite, [50, 90]).tolist()
                                                   if len(finite) else None),
                  joint_counts={str(v): int(np.count_nonzero(associated & (errors <= v)))
                                for v in DEPTH_BINS_M},
                  joint_denominator="all_nonzero_source_depth_points")
    return result, associated, errors


def run(args):
    plan = json.loads(args.plan.read_text())
    native = json.loads(args.native_report.read_text())
    if (native.get("schema") != "ms2_native_panel_inventory_v1"
            or native["frame_ids"] != plan["frame_ids"] or native["sequence"] != plan["sequence"]
            or native["calibration_numerical_errors"] or native["calibration_required_missing"]
            or native["stereo_depth_dimension_errors"]):
        raise ValueError("native panel prerequisites failed")
    for path in (args.plan, args.sync_report, args.depth_report):
        if native["input_sha256"].get(str(path)) != file_sha256(path):
            raise ValueError("native report inputs changed")
    files = {**verified_files(args.sync_root, args.sync_report, args.plan, "sync_data"),
             **verified_files(args.depth_root, args.depth_report, args.plan, "proj_depth")}
    if {k: file_sha256(v) for k, v in files.items()} != native["extracted_sha256"]:
        raise ValueError("native extracted identities changed")
    seq = plan["sequence"]
    c = read_calibration(files[f"sync_data/{seq}/calib.npy"].read_bytes())
    fixed = via_common_camera(from_millimetres(c["R_nir2thr"], c["T_nir2thr"]),
                              from_millimetres(c["R_nir2rgb"], c["T_nir2rgb"]))
    times = {s: parse_timestamps(files[f"sync_data/{seq}/{s}/img_left_timestamp.txt"].read_bytes(),
                                 expected_count=plan["frame_count"]) for s in ("rgb", "thr")}
    poses = read_rgb_poses(args.odom, plan)
    rows = []
    for frame in plan["frame_ids"]:
        i = int(frame)
        row = dict(frame=frame, thermal_minus_rgb_ns=int(times["thr"][i] - times["rgb"][i]))
        transforms = {"static": fixed}
        try:
            target_pose, info = interpolate_pose(poses, times["rgb"], int(times["thr"][i]),
                                                 max_extrapolation_ns=20_000_000)
            row["pose_interpolation"] = info
            row["strict_interpolation_available"] = not info["extrapolated"]
            transforms["ego_bounded_20ms"] = moving_rig_transform(fixed, poses[i], target_pose)
            transforms["inverse_motion_control"] = moving_rig_transform(
                fixed, target_pose, poses[i])
        except ValueError as error:
            row["motion_unavailable"] = str(error)
            row["strict_interpolation_available"] = False
        depths = {}
        for sensor in ("rgb", "thr"):
            with Image.open(files[f"proj_depth/{seq}/{sensor}/depth/{frame}.png"]) as image:
                depths[sensor] = np.array(image)
        row["directions"] = {}
        for source, target in (("rgb", "thr"), ("thr", "rgb")):
            measured = {}
            for name, transform in transforms.items():
                if source == "thr":
                    transform = np.linalg.inv(transform)
                measured[name] = measure(depths[source], depths[target], c[f"K_{source}L"],
                                         c[f"K_{target}L"], transform)
            results = {name: value[0] for name, value in measured.items()}
            for name in set(measured) - {"static"}:
                common = measured["static"][1] & measured[name][1]
                results[name]["common_association_with_static"] = dict(
                    count=int(common.sum()),
                    static_median_m=(float(np.median(measured["static"][2][common]))
                                     if common.any() else None),
                    candidate_median_m=(float(np.median(measured[name][2][common]))
                                        if common.any() else None))
            row["directions"][f"{source}_to_{target}"] = results
        rows.append(row)
        print(f"measured {frame}; strict interpolation={row['strict_interpolation_available']}",
              flush=True)
    return dict(
        schema="ms2_ego_motion_probe_v1", sequence=seq, rows=rows,
        protocol=dict(association_radius_px=1., depth_report_bins_m=DEPTH_BINS_M,
                      depth_convention="camera_axial_z_m", pose_reference="RGB trajectory only",
                      pose_clock_hypothesis="RGB odometry indexed at RGB-left timestamp",
                      max_extrapolation_ns=20_000_000,
                      interpolation="linear_translation_shortest_rotation",
                      transforms="T_fixed @ inverse(P_rgb_at_thermal_time) @ P_rgb_at_rgb_time",
                      all_selected_frames_retained=True, same_lidar_return_identity_verified=False,
                      independently_moving_objects_compensated=False,
                      reference_calibration_independent_of_depth_products=False,
                      exploratory_not_preregistered_qualification=True),
        input_sha256={str(p): file_sha256(p) for p in (
            args.plan, args.sync_report, args.depth_report, args.native_report, args.odom)},
        registration_qualified=False, generator_training_approved=False,
        antiuav_status_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(plan="ms2_train_screen_plan_01/plan.json",
                    sync_root="ms2_train_screen_sync_01", depth_root="ms2_train_screen_depth_01",
                    sync_report="ms2_train_screen_sync_01_report.json",
                    depth_report="ms2_train_screen_depth_01_report.json",
                    native_report="ms2_native_screen_01/report.json",
                    odom="external/ms2_first_train_archives/odom.prefix.tar.bz2")
    for name, value in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            default=Path("experiments") / value)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(args)
    root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {name: file_sha256(root / name) for name in (
        "scripts/probe_ms2_ego_motion.py", "scripts/audit_ms2_screen.py",
        "src/aero_ir/data/ms2_archive.py", "src/aero_ir/data/ms2_calibration.py",
        "src/aero_ir/data/ms2_timestamps.py", "src/aero_ir/registration/calibrated.py",
        "src/aero_ir/registration/ego_motion.py", "src/aero_ir/utils/manifest.py")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out}; no image-registration qualification")


if __name__ == "__main__":
    main()
