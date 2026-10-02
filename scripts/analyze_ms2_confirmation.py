"""Recompute confirmation scores and retain every planned frame and candidate."""
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
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_confirmation import candidate_scores
from scripts.probe_ms2_ego_motion import read_rgb_poses


def compare_values(actual, expected, label="scores"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"different score keys: {label}")
        for key in expected:
            compare_values(actual[key], expected[key], f"{label}.{key}")
    elif expected is None:
        if actual is not None:
            raise ValueError(f"expected missing statistic: {label}")
    elif (not isinstance(actual, (int, float))
          or not np.isclose(actual, expected, atol=1e-12, rtol=0)):
        raise ValueError(f"saved score disagrees with reconstruction: {label}")


def aggregate(rows, frames, candidates):
    expected = {(model, frame, condition) for model in ("xoftr_640", "minima_xoftr")
                for frame in frames for condition in ("paired", "unrelated_thermal")}
    keys = [(r["model"], r["frame"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("incomplete or duplicate planned confirmation cases")
    if any(set(row["scores"]) != set(candidates) for row in rows):
        raise ValueError("missing or unexpected candidate")
    for row in rows:
        counts = {s["fixed_supported"]["points"] for s in row["scores"].values()}
        all_counts = {s["all_matches"]["points"] for s in row["scores"].values()}
        if len(counts) != 1 or len(all_counts) != 1:
            raise ValueError("candidate denominators differ")
    result = []
    for model in ("xoftr_640", "minima_xoftr"):
        for condition in ("paired", "unrelated_thermal"):
            items = {r["frame"]: r for r in rows
                     if r["model"] == model and r["condition"] == condition}
            for candidate in candidates:
                stats = [items[f]["scores"][candidate] for f in frames]
                fixed = [s["fixed_supported"] for s in stats]
                fractions = [s["fraction_all"]["3.0"] for s in fixed]
                base = [items[f]["scores"]["author_calibration"]["fixed_supported"]
                        ["fraction_all"]["3.0"] for f in frames]
                deltas = [a-b if a is not None and b is not None else None
                          for a, b in zip(fractions, base, strict=True)]
                finite_deltas = [d for d in deltas if d is not None]
                medians = [s["conditional_median_px"] for s in fixed
                           if s["conditional_median_px"] is not None]
                result.append(dict(model=model, condition=condition, candidate=candidate,
                                   planned_frames=len(frames),
                                   frames_without_reference_support=sum(
                                       s["points"] == 0 for s in fixed),
                                   all_image_matches=sum(s["all_matches"]["points"] for s in stats),
                                   fixed_reference_points=sum(s["points"] for s in fixed),
                                   candidate_projection_failures=sum(
                                       s["points"]-s["supported"] for s in fixed),
                                   equal_frame_score_missing_as_zero=float(np.mean(
                                       [v if v is not None else 0. for v in fractions])),
                                   median_of_conditional_frame_medians_px=float(np.median(medians))
                                   if medians else None,
                                   improved_frames=sum(d > 1e-12 for d in finite_deltas),
                                   worsened_frames=sum(d < -1e-12 for d in finite_deltas),
                                   tied_frames=sum(abs(d) <= 1e-12 for d in finite_deltas),
                                   incomparable_frames=len(frames)-len(finite_deltas),
                                   frame_pck3=dict(zip(frames, fractions, strict=True)),
                                   frame_pck3_delta=dict(zip(frames, deltas, strict=True))))
    return result


def analyze(report_path):
    report = json.loads(report_path.read_text())
    if (report.get("schema") != "ms2_new_train_frame_confirmation_v1"
            or report.get("calibration_refitted") is not False):
        raise ValueError("unexpected or refitted confirmation report")
    preflight_path = report_path.parent / "preflight.json"
    if file_sha256(preflight_path) != report["preflight_sha256"]:
        raise ValueError("preflight changed")
    preflight = json.loads(preflight_path.read_text())
    plan_path = Path("experiments/ms2_confirmation_plan_01/plan.json")
    if file_sha256(plan_path) != preflight["plan_sha256"]:
        raise ValueError("confirmation plan changed")
    hashes = report["input_and_source_sha256"].copy()
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"dependency changed: {path}")
    plan = json.loads(plan_path.read_text())
    config = yaml.safe_load(Path("configs/experiment/registration_ms2_image_cpu.yaml").read_text())
    base = json.loads(Path(config["plan"]).read_text())
    root = Path("experiments/ms2_confirmation_sync_01/sync_data") / plan["sequence"]
    calib = read_calibration((root / "calib.npy").read_bytes())
    fixed = via_common_camera(from_millimetres(calib["R_nir2thr"], calib["T_nir2thr"]),
                              from_millimetres(calib["R_nir2rgb"], calib["T_nir2rgb"]))
    times = {s: parse_timestamps((root / s / "img_left_timestamp.txt").read_bytes(),
                                 expected_count=base["frame_count"]) for s in ("rgb", "thr")}
    poses = read_rgb_poses(Path(config["odom"]), base)
    fb = calib["K_rgbL"][0, 0] * -calib["T_rgbR"].reshape(3)[0] / 1000.
    transforms, disparities = {}, {}
    frame_rows = {row["frame"]: row for row in report["frames"]}
    if len(frame_rows) != len(report["frames"]) or set(frame_rows) != set(plan["frame_ids"]):
        raise ValueError("incomplete frame geometry")
    for frame in plan["frame_ids"]:
        row = frame_rows[frame]
        path = report_path.parent / row["stereo_file"]
        if path.parent != report_path.parent or file_sha256(path) != row["stereo_sha256"]:
            raise ValueError("changed or unsafe stereo artifact")
        hashes[str(path)] = row["stereo_sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            disparities[frame] = dict(arrays)
        pose, _ = interpolate_pose(poses, times["rgb"], int(times["thr"][int(frame)]),
                                   max_extrapolation_ns=20000000)
        transforms[frame] = moving_rig_transform(fixed, poses[int(frame)], pose)
    for row in report["rows"]:
        frame = row["frame"]
        index = plan["frame_ids"].index(frame)
        target_frame = frame if row["condition"] == "paired" else plan["frame_ids"][(index+7) % 15]
        if row["target_frame"] != target_frame:
            raise ValueError("wrong target-frame pairing")
        path = report_path.parent / row["artifact"]
        if path.parent != report_path.parent or file_sha256(path) != row["sha256"]:
            raise ValueError("changed or unsafe correspondence artifact")
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            disparity, _ = sample_disparity(disparities[frame], arrays["source_xy"])
            depth = fb/disparity
            if not np.allclose(depth, arrays["stereo_depth_m"], atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError("saved depth differs from image-stereo support")
            scores, eligible = candidate_scores(
                arrays["source_xy"], arrays["target_xy"], depth,
                calib["K_rgbL"], calib["K_thrL"], transforms[frame], plan["candidate_transforms"])
            if not np.array_equal(eligible, arrays["author_supported"]):
                raise ValueError("fixed support mask differs")
        compare_values(row["scores"], scores)
    return dict(schema="ms2_confirmation_verified_analysis_v1",
                aggregate=aggregate(report["rows"], plan["frame_ids"],
                                    plan["candidate_transforms"]),
                report_sha256=file_sha256(report_path), verified_artifact_sha256=hashes,
                source_sha256=file_sha256(__file__), registration_qualified=False,
                generator_training_approved=False,
                verification_scope="saved-data score reconstruction, NOT model replay or pixel GT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = analyze(args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out}; candidate confirmation is not registration qualification")


if __name__ == "__main__":
    main()
