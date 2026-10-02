"""Repeat bounded rig correction with RGB-only stereo depth; exploratory, not qualification."""
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
from scripts.probe_ms2_calibration_refinement import fit, score
from scripts.probe_ms2_ego_motion import read_rgb_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/experiment/registration_ms2_image_cpu.yaml").read_text())
    report_path = Path("experiments/ms2_stereo_depth_01/report.json")
    expected = "1b743c90200668914df30efbe90610af803be3340d21e1135027e45be101d5a4"
    if file_sha256(report_path) != expected:
        raise ValueError("stereo-depth report changed")
    report = json.loads(report_path.read_text())
    hashes = report["input_and_source_sha256"].copy()
    for path in (report_path, Path(__file__), Path("scripts/probe_ms2_calibration_refinement.py")):
        hashes[str(path)] = file_sha256(path)
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
        if row["condition"] != "paired":
            continue
        frame = row["frame"]
        path = report_path.parent / row["artifact"]
        if path.parent != report_path.parent or file_sha256(path) != row["sha256"]:
            raise ValueError("modified or unsafe stereo-depth artifact")
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            a, b, depth = data["source_xy"], data["target_xy"], data["stereo_depth_m"]
            # Availability is from stereo LR/support + geometric projection, not error magnitude.
            supported = np.isfinite(data["stereo_error"])
        points = (np.c_[a[supported], np.ones(np.count_nonzero(supported))]
                  @ np.linalg.inv(calib["K_rgbL"]).T) * depth[supported, None]
        pose, _ = interpolate_pose(poses, times["rgb"], int(times["thr"][int(frame)]),
                                   max_extrapolation_ns=config["pose_max_extrapolation_ns"])
        transform = moving_rig_transform(fixed, poses[int(frame)], pose)
        current_thermal = points @ transform[:3, :3].T + transform[:3, 3]
        cases.append(dict(model=row["model"], frame=frame,
                          split="fit" if frame in panels["fit"] else "check",
                          source_matches=len(a), points=current_thermal, observed=b[supported]))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(input_and_source_sha256=hashes, panels=panels,
                     bounds_per_axis=dict(rotation_degrees=3., translation_m=.1),
                     candidate_models=["rotation_only", "rigid6"],
                     loss="soft_l1", loss_scale_target_px=1., fit_weight="equal per match",
                     note="same prior8/8 exploratory train split, not unseen official test",
                     calibration_overwrite=False)
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    candidates = [dict(name="author_calibration", transform=np.eye(4).tolist())]
    for model in config["models"]:
        training = [x for x in cases if x["model"] == model and x["split"] == "fit"]
        for kind, dimensions in (("rotation_only", 3), ("rigid6", 6)):
            candidate = fit(np.concatenate([x["points"] for x in training]),
                            np.concatenate([x["observed"] for x in training]),
                            calib["K_thrL"], dimensions)
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
            raise ValueError(f"dependency changed during fit: {path}")
    result = dict(schema="ms2_stereo_depth_calibration_diagnostic_v1", rows=rows,
                  candidates=candidates, input_and_source_sha256=hashes,
                  preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
                  registration_qualified=False, generator_training_approved=False,
                  note="RGB stereo depth estimate; bounded calibration candidate, not pixel GT",
                  native_calibration_overwritten=False)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
