"""Bounded thermal-intrinsic sensitivity after frozen rotation; exploratory only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import (
    from_millimetres,
    intrinsic_matrix,
    project_pixels,
    via_common_camera,
)
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_confirmation import analyze
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_ms2_stereo_depth import summarize


def corrected_intrinsics(k, parameters):
    """Four normalized parameters: ±2% focal and ±3 native-pixel principal point."""
    k = intrinsic_matrix(k).copy()
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (4,) or not np.isfinite(parameters).all():
        raise ValueError('expected four finite normalized parameters')
    k[0, 0] *= 1+.02*parameters[0]
    k[1, 1] *= 1+.02*parameters[1]
    k[0, 2] += 3*parameters[2]
    k[1, 2] += 3*parameters[3]
    return intrinsic_matrix(k)


def fit_intrinsics(points, observed, k):
    """Fit positive-Z camera points without per-point/frame rejection or new pose fit."""
    if (points.shape != (len(observed), 3) or observed.shape != (len(points), 2)
            or len(points) < 4 or not np.isfinite(points).all()
            or not np.isfinite(observed).all() or np.any(points[:, 2] <= 0)):
        raise ValueError('finite positive-Z camera points and observations required')
    def residual(parameters):
        projected = points @ corrected_intrinsics(k, parameters).T
        return (projected[:, :2]/projected[:, 2, None]-observed).ravel()
    result = least_squares(residual, np.zeros(4), bounds=(-1., 1.),
                           loss='soft_l1', f_scale=1., max_nfev=1000)
    return dict(parameters=result.x.tolist(), intrinsic=corrected_intrinsics(k, result.x).tolist(),
                optimizer_success=bool(result.success), evaluations=int(result.nfev),
                bound_hit=(np.abs(result.x) >= .999).tolist(), fit_points=len(points))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    confirmation_path = Path('experiments/ms2_confirmation_image_01/report.json')
    verified = analyze(confirmation_path)
    hashes = verified['verified_artifact_sha256'].copy()
    config_path = Path('configs/experiment/registration_ms2_image_cpu.yaml')
    config = yaml.safe_load(config_path.read_text())
    plan_path = Path('experiments/ms2_confirmation_plan_01/plan.json')
    plan = json.loads(plan_path.read_text())
    base = json.loads(Path(config['plan']).read_text())
    original_path = Path('experiments/ms2_stereo_depth_01/report.json')
    for path in (original_path, confirmation_path, config_path, plan_path, Path(__file__),
                 Path('scripts/analyze_ms2_confirmation.py')):
        digest = file_sha256(path)
        if str(path) in hashes and hashes[str(path)] != digest:
            raise ValueError(f'changed frozen dependency: {path}')
        hashes[str(path)] = digest
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'changed dependency: {path}')
    original = json.loads(original_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    root = Path(config['sync_root'])/'sync_data'/plan['sequence']
    calib = read_calibration((root/'calib.npy').read_bytes())
    k = calib['K_thrL']
    fixed = via_common_camera(from_millimetres(calib['R_nir2thr'], calib['T_nir2thr']),
                              from_millimetres(calib['R_nir2rgb'], calib['T_nir2rgb']))
    times = {s: parse_timestamps((root/s/'img_left_timestamp.txt').read_bytes(),
                                 expected_count=base['frame_count']) for s in ('rgb', 'thr')}
    poses = read_rgb_poses(Path(config['odom']), base)
    fit_ids, check_ids = base['frame_ids'][::2], base['frame_ids'][1::2]
    cases = []
    for report, parent, is_original in ((original, original_path.parent, True),
                                        (confirmation, confirmation_path.parent, False)):
        for row in report['rows']:
            # Both reports include wrong-pair controls; these never enter fitting.
            path = parent/row['artifact']
            if path.parent != parent or file_sha256(path) != row['sha256']:
                raise ValueError('changed or unsafe match artifact')
            hashes[str(path)] = row['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                source, target, depth = (arrays[key] for key in
                                         ('source_xy', 'target_xy', 'stereo_depth_m'))
            frame = row['frame']
            pose, _ = interpolate_pose(poses, times['rgb'], int(times['thr'][int(frame)]),
                                       max_extrapolation_ns=20000000)
            transform = moving_rig_transform(fixed, poses[int(frame)], pose)
            projection = project_pixels(source, depth, calib['K_rgbL'], k, transform,
                                        source_shape=(384, 1224), target_shape=(256, 640))
            split = ('fit8' if frame in fit_ids else 'previous_check8') if is_original else (
                'observed_confirmation15')
            cases.append(dict(model=row['model'], frame=frame, condition=row['condition'],
                              split=split, source=source, target=target, depth=depth,
                              transform=transform, support=projection.supported))
    # Freeze the explicit perturbation range and fit/check IDs before any optimization.
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(input_and_source_sha256=hashes, fit_ids=fit_ids, check_ids=check_ids,
                     observed_confirmation_ids=plan['frame_ids'],
                     focal_fraction_bound=.02, principal_point_pixel_bound=3.,
                     loss='soft_l1', loss_scale_px=1., fit_weight='equal per match',
                     rotations_refitted=False, official_calibration_overwritten=False,
                     note='Bounds are diagnostic perturbations, not measured uncertainty. '
                     'All evaluation panels have already been observed; no fresh confirmation.')
    (args.out_dir/'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
    candidates = [dict(name=name, rotation=correction, intrinsic=k.tolist())
                  for name, correction in plan['candidate_transforms'].items()]
    for model in ('xoftr_640', 'minima_xoftr'):
        correction = np.asarray(plan['candidate_transforms'][f'{model}_rotation_only'])
        points, targets = [], []
        for case in cases:
            if (case['model'], case['split'], case['condition']) != (model, 'fit8', 'paired'):
                continue
            keep = case['support']
            source = (np.c_[case['source'][keep], np.ones(keep.sum())]
                      @ np.linalg.inv(calib['K_rgbL']).T) * case['depth'][keep, None]
            transform = correction @ case['transform']
            points.append(source @ transform[:3, :3].T + transform[:3, 3])
            targets.append(case['target'][keep])
        fit = fit_intrinsics(np.concatenate(points), np.concatenate(targets), k)
        candidates.append(dict(name=f'{model}_rotation_plus_intrinsic',
                               rotation=correction.tolist(), **fit))
    rows = []
    for candidate in candidates:
        for case in cases:
            projection = project_pixels(case['source'], case['depth'], calib['K_rgbL'],
                                        np.asarray(candidate['intrinsic']),
                                        np.asarray(candidate['rotation']) @ case['transform'],
                                        source_shape=(384, 1224), target_shape=(256, 640))
            errors = np.linalg.norm(case['target']-projection.target_xy, axis=1)
            errors[~case['support'] | ~projection.supported] = np.inf
            rows.append({key: case[key] for key in ('model', 'frame', 'condition', 'split')}
                        | dict(candidate=candidate['name'], all_matches=summarize(errors),
                               fixed_supported=summarize(errors[case['support']])))
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'dependency changed during fitting: {path}')
    report = dict(schema='ms2_intrinsic_sensitivity_v1', candidates=candidates, rows=rows,
                  preflight_sha256=file_sha256(args.out_dir/'preflight.json'),
                  input_and_source_sha256=hashes, registration_qualified=False,
                  generator_training_approved=False,
                  note='Post-confirmation exploratory hypothesis, not a new held-out result. '
                  'Target intrinsics may absorb matcher/system errors; not physical calibration.')
    (args.out_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'wrote {args.out_dir / "report.json"}; exploratory, not qualification')


if __name__ == '__main__':
    main()
