"""Attribute residual geometry without refitting or removing failed correspondences.

Epipolar distance is an optimistic lower bound over arbitrary source depth.
The disparity interval is a sensitivity probe, NOT a calibrated uncertainty bound.
Neither optimistic bound is a corrected match or qualification evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, project_pixels, via_common_camera
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_confirmation import analyze
from scripts.analyze_ms2_epipolar import fundamental_matrix
from scripts.probe_ms2_ego_motion import read_rgb_poses


def residual_components(source, target, projected, source_k, target_k, transform):
    """Signed orthogonal components in native target pixels; invalid stays NaN."""
    matrix = fundamental_matrix(source_k, target_k, transform)
    lines = np.c_[source, np.ones(len(source))] @ matrix.T
    norm = np.linalg.norm(lines[:, :2], axis=1)
    valid = (np.isfinite(lines).all(axis=1) & (norm > 1e-12)
             & np.isfinite(projected).all(axis=1) & np.isfinite(target).all(axis=1))
    normal = np.full((len(source), 2), np.nan)
    normal[valid] = lines[valid, :2] / norm[valid, None]
    tangent = np.c_[-normal[:, 1], normal[:, 0]]
    residual = target - projected
    return np.sum(residual * normal, axis=1), np.sum(residual * tangent, axis=1)


def bounded_disparity_error(source, target, disparity, fb, source_k, target_k,
                            transform, radius=1.):
    """Best possible residual over [max(0,d-radius),d+radius].

Projection is a rational function of disparity with a straight-line image locus.
When both endpoint depths are in front of the camera, the locus is exactly the
segment between endpoint projections. d=0 is the point at infinity. Bounds do
not restrict target image extent: this is deliberately optimistic. Invalid
intervals remain infinite error and are counted, never dropped.
"""
    if not np.isfinite(radius) or radius < 0 or not np.isfinite(fb) or fb <= 0:
        raise ValueError("finite nonnegative radius and positive focal-baseline required")
    disparity = np.asarray(disparity)
    rays = np.c_[source, np.ones(len(source))] @ np.linalg.inv(source_k).T
    rotated = (rays @ transform[:3, :3].T) * fb
    endpoints = []
    valid = (np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
             & np.isfinite(disparity) & (disparity > 0))
    for d in (np.maximum(0., disparity-radius), disparity+radius):
        points = rotated + d[:, None] * transform[:3, 3]
        projected = points @ target_k.T
        valid &= np.isfinite(projected).all(axis=1) & (projected[:, 2] > 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            endpoints.append(projected[:, :2] / projected[:, 2, None])
    start, stop = endpoints
    direction = stop-start
    squared = np.sum(direction**2, axis=1)
    weight = np.zeros(len(source))
    nonzero = valid & (squared > 1e-24)
    weight[nonzero] = np.clip(np.sum((target-start)[nonzero]*direction[nonzero], axis=1)
                              / squared[nonzero], 0., 1.)
    errors = np.full(len(source), np.inf)
    errors[valid] = np.linalg.norm((target-start-weight[:, None]*direction)[valid], axis=1)
    return errors


def summarize_mask(mask, errors, perpendicular, parallel, bounded):
    """Retain the supplied denominator even when a diagnostic value is invalid."""
    count = int(np.count_nonzero(mask))
    finite = mask & np.isfinite(errors)
    failed = mask & (~np.isfinite(errors) | (errors > 3.))
    cross_bad = failed & np.isfinite(perpendicular) & (np.abs(perpendicular) > 3.)
    def median(value):
        values = value[mask & np.isfinite(value)]
        return float(np.median(values)) if len(values) else None
    return dict(points=count, invalid_projection=int(np.count_nonzero(mask & ~np.isfinite(errors))),
                pck3=float(np.count_nonzero(finite & (errors <= 3.))/count) if count else None,
                median_error_px=median(errors), median_abs_cross_line_px=median(abs(perpendicular)),
                median_abs_along_line_px=median(abs(parallel)),
                failed_pck3=int(failed.sum()),
                failures_not_fixable_by_depth_alone=int(cross_bad.sum()),
                invalid_line_geometry=int(np.count_nonzero(mask & ~np.isfinite(perpendicular))),
                optimistic_disparity_radius1_pck3=float(np.count_nonzero(
                    mask & np.isfinite(bounded) & (bounded <= 3.))/count) if count else None,
                invalid_disparity_interval=int(np.count_nonzero(mask & ~np.isfinite(bounded))))


def strata(depth, confidence, source):
    """Fixed descriptive bins; these do not select a training/qualification subset."""
    result = {"all": np.ones(len(depth), bool)}
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, np.inf)):
        result[f"depth_m_{lo}_{hi}"] = (depth >= lo) & (depth < hi)
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, np.inf)):
        result[f"confidence_{lo}_{hi}"] = (confidence >= lo) & (confidence < hi)
    for iy in range(2):
        for ix in range(3):
            result[f"source_cell_{ix}_{iy}"] = (
                (source[:, 0] >= ix*408) & (source[:, 0] < (ix+1)*408)
                & (source[:, 1] >= iy*192) & (source[:, 1] < (iy+1)*192))
    return result


def aggregate(rows):
    result = []
    groups = sorted({(r['model'], r['condition'], r['candidate']) for r in rows})
    for model, condition, candidate in groups:
        cases = [r for r in rows if (r['model'], r['condition'], r['candidate'])
                 == (model, condition, candidate)]
        for key in cases[0]['strata']:
            stats = [r['strata'][key] for r in cases]
            points = sum(s['points'] for s in stats)
            present = [s for s in stats if s['points']]
            counts = {k: sum(s[k] for s in stats) for k in (
                'invalid_projection', 'failed_pck3', 'failures_not_fixable_by_depth_alone',
                'invalid_line_geometry', 'invalid_disparity_interval')}
            scores = {k: float(np.mean([s[k] for s in present])) if present else None
                      for k in ('pck3', 'optimistic_disparity_radius1_pck3')}
            medians = {}
            for metric in ('median_error_px', 'median_abs_cross_line_px',
                           'median_abs_along_line_px'):
                values = [s[metric] for s in stats if s[metric] is not None]
                medians[metric] = float(np.median(values)) if values else None
            result.append(dict(model=model, condition=condition, candidate=candidate, stratum=key,
                               planned_frames=len(cases), represented_frames=len(present),
                               points=points, counts=counts, equal_represented_frame_scores=scores,
                               median_of_frame_medians=medians))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=Path("experiments/ms2_confirmation_image_01/report.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    verified = analyze(args.report)
    report = json.loads(args.report.read_text())
    plan_path = Path("experiments/ms2_confirmation_plan_01/plan.json")
    plan = json.loads(plan_path.read_text())
    config_path = Path("configs/experiment/registration_ms2_image_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    base = json.loads(Path(config["plan"]).read_text())
    root = Path("experiments/ms2_confirmation_sync_01/sync_data") / plan["sequence"]
    calib = read_calibration((root/"calib.npy").read_bytes())
    fixed = via_common_camera(from_millimetres(calib["R_nir2thr"], calib["T_nir2thr"]),
                              from_millimetres(calib["R_nir2rgb"], calib["T_nir2rgb"]))
    times = {s: parse_timestamps((root/s/"img_left_timestamp.txt").read_bytes(),
                                 expected_count=base["frame_count"]) for s in ("rgb", "thr")}
    poses = read_rgb_poses(Path(config["odom"]), base)
    fb = calib["K_rgbL"][0, 0] * -calib["T_rgbR"].reshape(3)[0] / 1000.
    transforms = {}
    for frame in plan["frame_ids"]:
        pose, _ = interpolate_pose(poses, times["rgb"], int(times["thr"][int(frame)]),
                                   max_extrapolation_ns=20000000)
        transforms[frame] = moving_rig_transform(fixed, poses[int(frame)], pose)
    rows = []
    for row in report["rows"]:
        with np.load(args.report.parent/row["artifact"], allow_pickle=False) as arrays:
            source, target = arrays["source_xy"], arrays["target_xy"]
            depth, confidence = arrays["stereo_depth_m"], arrays["confidence"]
            supported = arrays["author_supported"]
        for name, correction in plan["candidate_transforms"].items():
            transform = np.asarray(correction) @ transforms[row["frame"]]
            projection = project_pixels(source, depth, calib["K_rgbL"], calib["K_thrL"],
                                        transform, source_shape=(384, 1224),
                                        target_shape=(256, 640))
            errors = np.linalg.norm(target-projection.target_xy, axis=1)
            errors[~projection.supported] = np.inf
            perpendicular, parallel = residual_components(
                source, target, projection.target_xy, calib["K_rgbL"], calib["K_thrL"], transform)
            bounded = bounded_disparity_error(source, target, fb/depth, fb, calib["K_rgbL"],
                                              calib["K_thrL"], transform)
            masks = strata(depth, confidence, source)
            rows.append({k: row[k] for k in ("model", "frame", "condition")}
                        | dict(candidate=name, all_matches=len(source),
                               reference_unsupported=int(np.count_nonzero(~supported)),
                               strata={key: summarize_mask(mask & supported, errors, perpendicular,
                                                           parallel, bounded)
                                       for key, mask in masks.items()}))
    hashes = verified["verified_artifact_sha256"].copy()
    for p in (args.report, plan_path, config_path, Path(__file__),
              Path("scripts/analyze_ms2_confirmation.py"), Path("scripts/analyze_ms2_epipolar.py")):
        hashes[str(p)] = file_sha256(p)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"dependency changed during analysis: {p}")
    result = dict(schema="ms2_residual_attribution_v1", rows=rows, aggregate=aggregate(rows),
                  input_and_source_sha256=hashes, disparity_probe_radius_px=1.,
                  note="post-confirmation descriptive analysis; no refit, filtering or new GT; "
                  "wrong pairs deliberately use true-pair geometry; disparity probe is an "
                  "optimistic bound, not measured uncertainty or an applied correction",
                  registration_qualified=False, generator_training_approved=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out}; residual attribution is not qualification")


if __name__ == "__main__":
    main()
