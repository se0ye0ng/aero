"""Compare independently image-estimated RGB/thermal stereo depth on a fixed RGB grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, project_pixels, via_common_camera
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.registration.temporal_stereo import estimate_pair, rectification, sample_native_depth
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_confirmation import add_identity, read
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_ms2_intrinsic_confirmation import load_protocol


def depth_statistics(expected, observed, reference):
    expected, observed = np.asarray(expected), np.asarray(observed)
    reference = np.asarray(reference)
    valid = (reference & np.isfinite(expected) & (expected > 0)
             & np.isfinite(observed) & (observed > 0))
    error = np.full(len(expected), np.inf)
    error[valid] = np.abs(observed[valid]-expected[valid])/expected[valid]
    count, supported = int(reference.sum()), int(valid.sum())
    return dict(reference_points=count, thermal_depth_available=supported,
                available_fraction=supported/count if count else None,
                fraction_reference_within_10pct=float(np.count_nonzero(
                    reference & (error <= .1))/count) if count else None,
                fraction_available_within_10pct=float(np.mean(error[valid] <= .1))
                if supported else None,
                conditional_median_relative_error=float(np.median(error[valid]))
                if supported else None), error, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    plan, config, hashes, files, calib, fb, transforms, _ = load_protocol()
    report_path = Path('experiments/ms2_intrinsic_confirmation_image_01/report.json')
    if file_sha256(report_path) != (
            'e2b4e5ea6695204ee28a23349611de05ca4eecbbca5b815f46115136d4995d04'):
        raise ValueError('RGB stereo reference changed')
    reference = json.loads(report_path.read_text())
    frames = {x['frame']: x for x in reference['frames']}
    for path in (report_path, Path(__file__),
                 Path('src/aero_ir/registration/temporal_stereo.py')):
        add_identity(hashes, path)
    seq = plan['sequence']
    base = json.loads(Path(config['plan']).read_text())
    times = {name: parse_timestamps(files[f'sync_data/{seq}/{sensor}/img_{side}_timestamp.txt']
                                    .read_bytes(), expected_count=base['frame_count'])
             for name, sensor, side in (('rgb', 'rgb', 'left'), ('right', 'thr', 'right'))}
    poses = read_rgb_poses(Path(config['odom']), base)
    fixed = via_common_camera(from_millimetres(calib['R_nir2thr'], calib['T_nir2thr']),
                              from_millimetres(calib['R_nir2rgb'], calib['T_nir2rgb']))
    baseline = from_millimetres(calib['R_thrR'], calib['T_thrR'])
    if (not np.allclose(calib['K_thrL'], calib['K_thrR'], atol=1e-9, rtol=0)
            or not np.allclose(calib['R_thrR'], np.eye(3), atol=1e-9, rtol=0)):
        raise ValueError('unexpected thermal stereo calibration')
    cameras = {k: plan['candidates'][k] for k in (
        'author_calibration', 'xoftr_640_rotation_plus_intrinsic',
        'minima_xoftr_rotation_plus_intrinsic')}
    yy, xx = np.mgrid[8:384:16, 8:1224:16]
    source = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(input_and_source_sha256=hashes, cameras=cameras,
                     source_grid='native RGB step16 offset8', refit=False,
                     thermal_window_dn=plan['thermal_window_dn'],
                     conditions=['paired_static', 'paired_timed', 'unrelated_right_timed'],
                     disparity='same frozen SGBM128/block5/LR1px/four-neighbor rules',
                     rectification='alpha1, zero disparity, already undistorted inputs',
                     correction_scope='effective left intrinsics also applied to right camera; '
                     'the author physical stereo baseline is kept unchanged',
                     diagnostic_relative_depth_threshold=.1, qualification_threshold=False,
                     opencv_version=cv2.__version__)
    (args.out_dir/'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
    thermal = {}
    for frame in plan['frame_ids']:
        for side in ('left', 'right'):
            raw = read(files[f'sync_data/{seq}/thr/img_{side}/{frame}.png'])
            if raw.dtype != np.uint16 or raw.shape != (256, 640):
                raise ValueError('invalid native thermal image')
            thermal[frame, side] = display_thermal(raw, plan['thermal_window_dn'])
    rows, common_rows = [], []
    for index, frame in enumerate(plan['frame_ids']):
        path = report_path.parent/frames[frame]['stereo_file']
        if path.parent != report_path.parent or file_sha256(path) != frames[frame]['stereo_sha256']:
            raise ValueError('changed or unsafe RGB stereo artifact')
        add_identity(hashes, path)
        with np.load(path, allow_pickle=False) as stereo:
            disparity, _ = sample_disparity(dict(stereo), source)
        depth = fb/disparity
        author = project_pixels(source, depth, calib['K_rgbL'], calib['K_thrL'], transforms[frame],
                                 source_shape=(384, 1224), target_shape=(256, 640))
        keep = author.supported
        pose, _ = interpolate_pose(poses, times['rgb'], int(times['right'][int(frame)]),
                                   max_extrapolation_ns=0)
        at_right = moving_rig_transform(fixed, poses[int(frame)], pose)
        wrong = plan['frame_ids'][(index+7) % 15]
        for name, camera in cameras.items():
            correction = np.asarray(camera['transform'])
            k = np.asarray(camera['target_intrinsic'])
            left_transform = correction @ transforms[frame]
            right_transform = baseline @ correction @ at_right
            timed_relative = right_transform @ np.linalg.inv(left_transform)
            projection = project_pixels(source, depth, calib['K_rgbL'], k, left_transform,
                                        source_shape=(384, 1224), target_shape=(256, 640))
            eligible = keep & projection.supported
            comparisons = {}
            for condition, relative, right_frame in (
                ('paired_static', baseline, frame), ('paired_timed', timed_relative, frame),
                ('unrelated_right_timed', timed_relative, wrong),
            ):
                geometry = rectification(k, k, relative, (256, 640))
                stereo = estimate_pair(thermal[frame, 'left'],
                                       thermal[right_frame, 'right'], geometry)
                estimated = np.full(len(source), np.nan)
                estimated[eligible], _ = sample_native_depth(
                    stereo, geometry, projection.target_xy[eligible])
                scores, errors, supported = depth_statistics(projection.target_z_m, estimated, keep)
                artifact = args.out_dir/f'{name}_{frame}_{condition}.npz'
                np.savez_compressed(artifact, **stereo,
                                    **{f'geometry_{key}': value for key, value in geometry.items()},
                                    source_xy=source, source_depth_m=depth,
                                    projected_xy=projection.target_xy,
                                    expected_z=projection.target_z_m,
                                    estimated_z=estimated, reference_supported=keep,
                                    jointly_supported=supported)
                rows.append(dict(frame=frame, right_frame=right_frame, camera=name,
                                 condition=condition, artifact=artifact.name,
                                 sha256=file_sha256(artifact), scores=scores,
                                 grid_points=len(source),
                                 rectified_supported_pixels=int(stereo['valid'].sum()),
                                 relative_transform=relative.tolist()))
                comparisons[condition] = errors, supported
            shared = comparisons['paired_static'][1] & comparisons['paired_timed'][1]
            common_errors = {key: float(np.median(errors[shared])) if shared.any() else None
                             for key, (errors, _) in comparisons.items()
                             if key.startswith('paired_')}
            common_rows.append(dict(frame=frame, camera=name, common_points=int(shared.sum()),
                                    median_relative_error=common_errors))
        print(f'{frame}: static/timed thermal stereo + wrong-right controls complete', flush=True)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'input changed during cross-stereo: {path}')
    result = dict(schema='ms2_cross_stereo_depth_v1', rows=rows, common_support=common_rows,
                  input_and_source_sha256=hashes,
                  preflight_sha256=file_sha256(args.out_dir/'preflight.json'),
                  registration_qualified=False, generator_training_approved=False,
                  note='Two image-derived depth estimates, not physical GT. Fixed RGB author '
                  'reference support; missing thermal depth is failure in full-reference scores. '
                  'Motion assumes static scene and supplied timestamps/odometry. No camera refit.')
    with (args.out_dir/'report.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out_dir / "report.json"}; cross-stereo is not qualification')


if __name__ == '__main__':
    main()
