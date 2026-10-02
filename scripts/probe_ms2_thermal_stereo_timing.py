"""Measure the geometric consequence of asynchronous thermal stereo before using its depth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, project_pixels, via_common_camera
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_confirmation import add_identity
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_ms2_intrinsic_confirmation import load_protocol


def statistics(value):
    finite = np.asarray(value)[np.isfinite(value)]
    return dict(count=len(value), finite=len(finite),
                median=float(np.median(finite)) if len(finite) else None,
                p95=float(np.percentile(finite, 95)) if len(finite) else None,
                maximum=float(np.max(finite)) if len(finite) else None)


def stereo_transforms(left_transform, right_time_transform, right_from_left, correction):
    """Apply the physical stereo baseline after the common left-camera correction."""
    return dict(left=correction @ left_transform,
                simultaneous_right=right_from_left @ correction @ left_transform,
                actual_right=right_from_left @ correction @ right_time_transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    plan, config, hashes, files, calib, fb, transforms, _ = load_protocol()
    reference_path = Path('experiments/ms2_intrinsic_confirmation_image_01/report.json')
    expected = 'e2b4e5ea6695204ee28a23349611de05ca4eecbbca5b815f46115136d4995d04'
    if file_sha256(reference_path) != expected:
        raise ValueError('changed RGB-stereo reference')
    reference = json.loads(reference_path.read_text())
    for path in (reference_path, Path(__file__)):
        add_identity(hashes, path)
    seq = plan['sequence']
    base = json.loads(Path(config['plan']).read_text())
    timestamps = {name: parse_timestamps(files[f'sync_data/{seq}/{sensor}/img_{side}_timestamp.txt']
                                         .read_bytes(), expected_count=base['frame_count'])
                  for name, sensor, side in (('rgb', 'rgb', 'left'), ('left', 'thr', 'left'),
                                             ('right', 'thr', 'right'))}
    poses = read_rgb_poses(Path(config['odom']), base)
    fixed = via_common_camera(from_millimetres(calib['R_nir2thr'], calib['T_nir2thr']),
                              from_millimetres(calib['R_nir2rgb'], calib['T_nir2rgb']))
    right_from_left = from_millimetres(calib['R_thrR'], calib['T_thrR'])
    if (not np.allclose(calib['K_thrL'], calib['K_thrR'], atol=1e-9, rtol=0)
            or not np.allclose(calib['R_thrR'], np.eye(3), atol=1e-9, rtol=0)):
        raise ValueError('expected common rectified thermal intrinsics and rotation')
    yy, xx = np.mgrid[8:384:16, 8:1224:16]
    source = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
    frames = {x['frame']: x for x in reference['frames']}
    rows = []
    for frame in plan['frame_ids']:
        path = reference_path.parent/frames[frame]['stereo_file']
        if (path.parent != reference_path.parent
                or file_sha256(path) != frames[frame]['stereo_sha256']):
            raise ValueError('changed or unsafe RGB stereo artifact')
        add_identity(hashes, path)
        with np.load(path, allow_pickle=False) as stereo:
            disparity, _ = sample_disparity(dict(stereo), source)
        depth = fb/disparity
        pose, info = interpolate_pose(
            poses, timestamps['rgb'], int(timestamps['right'][int(frame)]), max_extrapolation_ns=0)
        right_time_transform = moving_rig_transform(fixed, poses[int(frame)], pose)
        for name, camera in plan['candidates'].items():
            correction = np.asarray(camera['transform'])
            k = np.asarray(camera['target_intrinsic'])
            projections = {}
            camera_transforms = stereo_transforms(transforms[frame], right_time_transform,
                                                   right_from_left, correction)
            for variant, transform in camera_transforms.items():
                projections[variant] = project_pixels(source, depth, calib['K_rgbL'], k, transform,
                                                       source_shape=(384, 1224),
                                                       target_shape=(256, 640))
            eligible = projections['left'].supported & projections['simultaneous_right'].supported
            valid = eligible & projections['actual_right'].supported
            delta = (projections['actual_right'].target_xy
                     - projections['simultaneous_right'].target_xy)
            stereo_shift = (projections['left'].target_xy[:, 0]
                            - projections['simultaneous_right'].target_xy[:, 0])
            measured_y = (projections['actual_right'].target_xy[:, 1]
                          - projections['left'].target_xy[:, 1])
            rows.append(dict(frame=frame, candidate=name,
                             thermal_right_minus_left_ns=int(timestamps['right'][int(frame)]
                                                             - timestamps['left'][int(frame)]),
                             grid_points=len(source), stereo_geometric_eligible=int(eligible.sum()),
                             temporal_projection_lost=int((eligible & ~valid).sum()),
                             displacement_px=statistics(np.linalg.norm(delta[valid], axis=1)),
                             absolute_vertical_mismatch_px=statistics(np.abs(measured_y[valid])),
                             signed_horizontal_shift_px=statistics(delta[valid, 0]),
                             relative_to_static_disparity=statistics(
                                 np.abs(delta[valid, 0]/stereo_shift[valid])),
                             right_time_pose_info=info))
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'input changed during timing analysis: {path}')
    result = dict(schema='ms2_thermal_stereo_timing_v1', rows=rows,
                  input_and_source_sha256=hashes, source_grid='native RGB, step16, offset8',
                  registration_qualified=False, generator_training_approved=False,
                  note='Static-scene, supplied-timestamp and RGB-odometry hypothesis. '
                  'This does not verify exposure timing or independently moving objects. '
                  'Thermal stereo depth is not computed or adopted here.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out}; thermal stereo timing diagnostic, not qualification')


if __name__ == '__main__':
    main()
