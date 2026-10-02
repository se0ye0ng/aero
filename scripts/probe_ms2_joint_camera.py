"""Joint bounded rotation/intrinsic fit versus frozen sequential fit; CPU diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, project_pixels, via_common_camera
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_calibration_refinement import ROTATION_SCALE, correction, predict
from scripts.probe_ms2_confirmation import add_identity
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_ms2_intrinsic_sensitivity import corrected_intrinsics
from scripts.probe_ms2_stereo_depth import summarize


def joint_predict(points, k, parameters):
    p = np.asarray(parameters, dtype=float)
    if p.shape != (7,) or not np.isfinite(p).all() or np.any(np.abs(p) > 1):
        raise ValueError("seven finite normalized parameters in [-1,1] required")
    return predict(points, corrected_intrinsics(k, p[3:]), correction(p[:3]))


def fit_joint(points, observed, k, start):
    if (
        points.shape != (len(observed), 3)
        or observed.shape != (len(points), 2)
        or len(points) < 12
        or not np.isfinite(points).all()
        or not np.isfinite(observed).all()
        or np.any(points[:, 2] <= 0)
    ):
        raise ValueError("finite positive-depth fit points required")

    def residual(p):
        return (joint_predict(points, k, p) - observed).ravel()

    starts = (np.zeros(7), np.asarray(start, dtype=float))
    fits = [
        least_squares(
            residual,
            s,
            bounds=(-1.0, 1.0),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=1000,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        for s in starts
    ]
    # Selection uses fit-set objective only, never check-frame scores.
    best = min(fits, key=lambda f: f.cost)
    return dict(
        parameters=best.x.tolist(),
        rotation=correction(best.x[:3]).tolist(),
        intrinsic=corrected_intrinsics(k, best.x[3:]).tolist(),
        success=bool(best.success),
        cost=float(best.cost),
        active_bounds=best.active_mask.tolist(),
        jacobian_singular_values=np.linalg.svd(best.jac, compute_uv=False).tolist(),
        initializations=[
            dict(success=bool(f.success), cost=float(f.cost), evaluations=f.nfev) for f in fits
        ],
        fit_points=len(points),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    sources = [
        (
            Path("experiments/ms2_stereo_depth_01/report.json"),
            "1b743c90200668914df30efbe90610af803be3340d21e1135027e45be101d5a4",
        ),
        (
            Path("experiments/ms2_intrinsic_confirmation_image_01/report.json"),
            "e2b4e5ea6695204ee28a23349611de05ca4eecbbca5b815f46115136d4995d04",
        ),
        (
            Path("experiments/ms2_intrinsic_sensitivity_01/report.json"),
            "d71ad7849fbc26bc1395b5475ed79978099bd148399a19a039df69c58b40e2ff",
        ),
    ]
    reports, hashes = [], {}
    for p, digest in sources:
        if file_sha256(p) != digest:
            raise ValueError(f"changed report: {p}")
        report = json.loads(p.read_text())
        reports.append(report)
        for key, value in report["input_and_source_sha256"].items():
            if key in hashes and hashes[key] != value:
                raise ValueError("conflicting frozen dependency")
            hashes[key] = value
        add_identity(hashes, p)
    config_path = Path("configs/experiment/registration_ms2_image_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    plan = json.loads(Path(config["plan"]).read_text())
    for p in (
        Path(__file__),
        config_path,
        Path(config["plan"]),
        Path("scripts/probe_ms2_calibration_refinement.py"),
        Path("scripts/probe_ms2_intrinsic_sensitivity.py"),
    ):
        add_identity(hashes, p)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed source: {p}")
    root = Path(config["sync_root"]) / "sync_data" / plan["sequence"]
    calib = read_calibration((root / "calib.npy").read_bytes())
    k = calib["K_thrL"]
    fixed = via_common_camera(
        from_millimetres(calib["R_nir2thr"], calib["T_nir2thr"]),
        from_millimetres(calib["R_nir2rgb"], calib["T_nir2rgb"]),
    )
    times = {
        s: parse_timestamps(
            (root / s / "img_left_timestamp.txt").read_bytes(), expected_count=plan["frame_count"]
        )
        for s in ("rgb", "thr")
    }
    poses = read_rgb_poses(Path(config["odom"]), plan)
    fit_ids = plan["frame_ids"][::2]
    cases = []
    for index, report in enumerate(reports[:2]):
        parent = sources[index][0].parent
        for r in report["rows"]:
            p = parent / r["artifact"]
            if p.resolve().parent != parent.resolve() or file_sha256(p) != r["sha256"]:
                raise ValueError("unsafe or changed match artifact")
            add_identity(hashes, p)
            with np.load(p, allow_pickle=False) as f:
                source, target, depth = (
                    f[key] for key in ("source_xy", "target_xy", "stereo_depth_m")
                )
            frame = r["frame"]
            pose, _ = interpolate_pose(
                poses,
                times["rgb"],
                int(times["thr"][int(frame)]),
                max_extrapolation_ns=20000000 if index == 0 else 0,
            )
            transform = moving_rig_transform(fixed, poses[int(frame)], pose)
            author = project_pixels(
                source,
                depth,
                calib["K_rgbL"],
                k,
                transform,
                source_shape=(384, 1224),
                target_shape=(256, 640),
            )
            split = (
                ("fit8" if frame in fit_ids else "check8") if index == 0 else "observed_quarter15"
            )
            support = author.supported
            points = (
                np.c_[source[support], np.ones(support.sum())] @ np.linalg.inv(calib["K_rgbL"]).T
            ) * depth[support, None]
            points = points @ transform[:3, :3].T + transform[:3, 3]
            cases.append(
                dict(
                    model=r["model"],
                    frame=frame,
                    condition=r["condition"],
                    split=split,
                    source=source,
                    target=target,
                    depth=depth,
                    transform=transform,
                    support=support,
                    camera_points=points,
                )
            )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(
        input_and_source_sha256=hashes,
        fit_ids=fit_ids,
        rotation_axis_bound_deg=3.0,
        focal_fraction_bound=0.02,
        principal_bound_px=3.0,
        loss="soft_l1",
        f_scale_px=1.0,
        initialization=["author_zero", "previous_sequential"],
        initialization_selection="minimum fit-set objective",
        frame_or_match_rejection=False,
        camera_installed=False,
        evaluation="all panels previously observed; exploratory, not held-out test",
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    candidates = list(reports[2]["candidates"])
    fits = {}
    for model in ("xoftr_640", "minima_xoftr"):
        sequential = next(c for c in candidates if c["name"] == f"{model}_rotation_plus_intrinsic")
        start = np.r_[
            Rotation.from_matrix(np.asarray(sequential["rotation"])[:3, :3]).as_rotvec()
            / ROTATION_SCALE,
            sequential["parameters"],
        ]
        selected = [
            c
            for c in cases
            if (c["model"], c["split"], c["condition"]) == (model, "fit8", "paired")
        ]
        fit = fit_joint(
            np.concatenate([c["camera_points"] for c in selected]),
            np.concatenate([c["target"][c["support"]] for c in selected]),
            k,
            start,
        )
        fits[model] = fit
        candidates.append(dict(name=f"{model}_joint_rotation_intrinsic", **fit))
        print(model, "fit complete", fit["success"], "bounds", fit["active_bounds"], flush=True)
    rows = []
    for candidate in candidates:
        for c in cases:
            projected = project_pixels(
                c["source"],
                c["depth"],
                calib["K_rgbL"],
                np.asarray(candidate["intrinsic"]),
                np.asarray(candidate["rotation"]) @ c["transform"],
                source_shape=(384, 1224),
                target_shape=(256, 640),
            )
            errors = np.linalg.norm(projected.target_xy - c["target"], axis=1)
            errors[~c["support"] | ~projected.supported] = np.inf
            rows.append(
                {key: c[key] for key in ("model", "frame", "split", "condition")}
                | dict(
                    candidate=candidate["name"],
                    all_matches=summarize(errors),
                    fixed_supported=summarize(errors[c["support"]]),
                )
            )
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed: {p}")
    result = dict(
        candidates=candidates,
        rows=rows,
        fits=fits,
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
        note="Effective camera fits image matches, not independent physical GT. "
        "All existing support retained; no source calibration overwritten.",
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}; not qualification")


if __name__ == "__main__":
    main()
