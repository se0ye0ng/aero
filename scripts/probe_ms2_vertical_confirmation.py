"""Fixed half-pixel sampling comparison on disjoint, previously observed quarter frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, ms2_depth_metres, via_common_camera
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.stereo_vertical_compensation import estimate
from aero_ir.registration.temporal_stereo import estimate_pair, rectification
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_cross_stereo import summarize_errors
from scripts.prepare_ms2_intrinsic_confirmation import PLAN_PATH, checked_plan
from scripts.prepare_ms2_vertical_confirmation import OUT, SOURCE, SOURCE_SHA
from scripts.probe_ms2_confirmation import add_identity
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_ms2_lidar_stereo import aggregate, timed_relative
from scripts.probe_ms2_raft_stereo import score
from scripts.probe_ms2_stereo_depth import read

OFFSETS = (0.0, 0.5, -0.5)


def validate_protocol(protocol, plan, selection_frames):
    if (
        protocol["frame_ids"] != plan["frame_ids"]
        or protocol["sequence"] != plan["sequence"]
        or protocol["source_sha256"] != SOURCE_SHA
        or protocol["offsets"] != list(OFFSETS)
        or protocol["wrong_pair_cyclic_offset"] != 7
        or protocol["thermal_window_dn"] != [3308.0, 4974.0]
        or protocol["camera_fit"]
        or protocol["registration_qualified"]
        or protocol["generator_training_approved"]
        or set(protocol["frame_ids"]) & set(selection_frames)
    ):
        raise ValueError("changed intervention protocol or overlapping panel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("frozen candidate evidence changed")
    plan = checked_plan(PLAN_PATH)
    source = json.loads(SOURCE.read_text())
    hashes = source["input_and_source_sha256"].copy()
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed source: {p}")
    sync = Path("experiments/ms2_intrinsic_confirmation_sync_01")
    for root, report in (
        (sync, sync.with_name(sync.name + "_report.json")),
        (OUT, OUT.with_name(OUT.name + "_report.json")),
    ):
        extraction = json.loads(report.read_text())
        if not extraction["complete"] or extraction["sequence"] != plan["sequence"]:
            raise ValueError("incomplete or wrong sequence extraction")
        expected_archive = (
            "c3a10f2cef0d04ea6999c9885311aa1bda206e3c228960ce3e1462d19e913902"
            if root == sync
            else "b148bda7f56ab0370bbb8dfbad05fd1a4b699b083186a4354dc9993e6a60f026"
        )
        if extraction["archive_sha256"] != expected_archive:
            raise ValueError("wrong extraction archive")
        add_identity(hashes, report)
        for entry in extraction["files"]:
            path = root / entry["path"]
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("unsafe extraction path")
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"changed extracted file: {path}")
            add_identity(hashes, path)
    protocol_path = OUT.with_name(OUT.name + "_plan.json")
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol, plan, [row["frame"] for row in source["rows"]])
    if extraction["plan_sha256"] != file_sha256(protocol_path):
        raise ValueError("extraction plan changed")
    preparation_snapshot = Path("scripts/prepare_ms2_vertical_confirmation.py")
    if protocol["preparation_source_sha256"] != file_sha256(preparation_snapshot):
        preparation_snapshot = Path("experiments/ms2_vertical_confirmation_preparation_frozen.py")
    if not preparation_snapshot.is_file() or protocol["preparation_source_sha256"] != file_sha256(
        preparation_snapshot
    ):
        raise ValueError("preparation source changed")
    add_identity(hashes, preparation_snapshot)
    for path in (
        SOURCE,
        PLAN_PATH,
        protocol_path,
        Path(__file__),
        Path("scripts/prepare_ms2_vertical_confirmation.py"),
    ):
        add_identity(hashes, path)
    base_plan = json.loads(Path("experiments/ms2_train_screen_plan_01/plan.json").read_text())
    root = sync / "sync_data" / plan["sequence"]
    calibration = read_calibration((root / "calib.npy").read_bytes())
    times = {
        (s, side): parse_timestamps(
            (root / s / f"img_{side}_timestamp.txt").read_bytes(),
            expected_count=plan["frame_count"],
        )
        for s in ("rgb", "thr")
        for side in ("left", "right")
    }
    poses = read_rgb_poses(
        Path("experiments/external/ms2_first_train_archives/odom.prefix.tar.bz2"), base_plan
    )
    fixed = dict(
        rgb=np.eye(4),
        thr=via_common_camera(
            from_millimetres(calibration["R_nir2thr"], calibration["T_nir2thr"]),
            from_millimetres(calibration["R_nir2rgb"], calibration["T_nir2rgb"]),
        ),
    )
    focal = float(calibration["K_thrL"][0, 0])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(
        protocol=protocol,
        input_and_source_sha256=hashes,
        equivalent_disparity_focal_px=focal,
        all_nonzero_lidar_points=True,
        opencv_version=cv2.__version__,
        conditions=["paired_static", "paired_timed", "unrelated_right_timed"],
        registration_qualified=False,
        generator_training_approved=False,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    cv2.setNumThreads(1)
    rows, common = [], []
    for sensor in ("rgb", "thr"):
        images = {}
        for frame in plan["frame_ids"]:
            for side in ("left", "right"):
                raw = read(root / sensor / f"img_{side}" / f"{frame}.png")
                images[frame, side] = (
                    cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
                    if sensor == "rgb"
                    else display_thermal(raw, (3308.0, 4974.0))
                )
        baseline = from_millimetres(calibration[f"R_{sensor}R"], calibration[f"T_{sensor}R"])
        for index, frame in enumerate(plan["frame_ids"]):
            depth = ms2_depth_metres(
                read(OUT / "proj_depth" / plan["sequence"] / sensor / "depth" / f"{frame}.png")
            )
            if depth.shape != images[frame, "left"].shape:
                raise ValueError("depth/image dimensions differ")
            yy, xx = np.nonzero(np.isfinite(depth))
            timed, pose_info = timed_relative(
                poses,
                times["rgb", "left"],
                int(times[sensor, "left"][int(frame)]),
                int(times[sensor, "right"][int(frame)]),
                fixed[sensor],
                baseline,
            )
            wrong = plan["frame_ids"][(index + 7) % len(plan["frame_ids"])]
            for condition, relative, right_frame in (
                ("paired_static", baseline, frame),
                ("paired_timed", timed, frame),
                ("unrelated_right_timed", timed, wrong),
            ):
                geometry = (
                    {}
                    if relative is None
                    else rectification(
                        calibration[f"K_{sensor}L"],
                        calibration[f"K_{sensor}R"],
                        relative,
                        depth.shape,
                    )
                )
                original = dict(
                    source_xy=np.c_[xx, yy].astype(float),
                    lidar_depth_m=depth[yy, xx],
                    equivalent_disparity_errors=np.full(len(xx), np.nan),
                    **{f"geometry_{k}": v for k, v in geometry.items()},
                )
                errors = {}
                for offset in OFFSETS:
                    stereo = (
                        estimate(
                            images[frame, "left"], images[right_frame, "right"], geometry, offset
                        )
                        if geometry
                        else {}
                    )
                    if offset == 0 and geometry:
                        old = estimate_pair(
                            images[frame, "left"], images[right_frame, "right"], geometry
                        )
                        for k, v in old.items():
                            np.testing.assert_array_equal(stereo[k], v)
                    z, e, scores = score(stereo, original, focal)
                    # The inherited score helper calls its candidate 'raft'. Do not
                    # retain that inapplicable baseline/RAFT comparison in this report.
                    scores.pop("common_support")
                    path = args.out_dir / f"{sensor}_{frame}_{condition}_dy{offset}.npz"
                    np.savez_compressed(
                        path,
                        **stereo,
                        **{k: v for k, v in original.items() if k != "equivalent_disparity_errors"},
                        estimated_depth_m=z,
                        equivalent_disparity_errors=e,
                    )
                    rows.append(
                        dict(
                            sensor=sensor,
                            frame=frame,
                            condition=condition,
                            right_frame=right_frame,
                            offset=offset,
                            geometry_available=bool(geometry),
                            timed_pose_info=pose_info,
                            artifact=path.name,
                            sha256=file_sha256(path),
                            **scores,
                        )
                    )
                    errors[str(offset)] = e
                mask = np.logical_and.reduce([np.isfinite(e) for e in errors.values()])
                common.append(
                    dict(
                        sensor=sensor,
                        frame=frame,
                        condition=condition,
                        all_reference_points=len(xx),
                        common_points=int(mask.sum()),
                        scores={k: summarize_errors(e, mask) for k, e in errors.items()},
                    )
                )
            print(
                f"{sensor} {frame}: all three conditions and sampling offsets complete", flush=True
            )
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during run: {p}")
    result = dict(
        schema="ms2_vertical_confirmation_v1",
        rows=rows,
        common_support=common,
        aggregate={str(o): aggregate([r for r in rows if r["offset"] == o]) for o in OFFSETS},
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}; not qualification", flush=True)


if __name__ == "__main__":
    main()
