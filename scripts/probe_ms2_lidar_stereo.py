"""Compare same-camera sparse LiDAR depth with image stereo, without fitting geometry.

Projected LiDAR inherits camera calibration and rasterization uncertainty. It is
an additional measurement, not independent dense RGB-to-IR correspondence GT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, ms2_depth_metres, via_common_camera
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.registration.temporal_stereo import (
    estimate_pair,
    native_to_rectified,
    rectification,
    sample_native_depth,
)
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_cross_stereo import expected_disparity, strata, summarize_errors
from scripts.probe_ms2_cross_stereo import depth_statistics
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_ms2_stereo_depth import read

SOURCE = Path('experiments/ms2_stereo_depth_01/report.json')
SOURCE_SHA = '1b743c90200668914df30efbe90610af803be3340d21e1135027e45be101d5a4'


def timed_relative(poses, rgb_times, left_ns, right_ns, left_from_rgb, baseline):
    """No extrapolation; return an explicit unavailable case at trajectory boundaries."""
    if min(left_ns, right_ns) < rgb_times[0] or max(left_ns, right_ns) > rgb_times[-1]:
        return None, dict(available=False, reason='exposure outside RGB trajectory coverage')
    left, li = interpolate_pose(poses, rgb_times, left_ns, max_extrapolation_ns=0)
    right, ri = interpolate_pose(poses, rgb_times, right_ns, max_extrapolation_ns=0)
    relative = moving_rig_transform(baseline @ left_from_rgb, left, right)
    return relative @ np.linalg.inv(left_from_rgb), dict(available=True, left=li, right=ri)


def summarize_by_depth(errors, depth):
    # Unlike image-match-derived support, reference is EVERY nonzero LiDAR pixel.
    # Reuse fixed 0/10/20/40 m bins, naming them explicitly as LiDAR depth here.
    return {key.replace('rgb_depth', 'lidar_depth'): summarize_errors(errors, mask)
            for key, mask in strata(depth).items()}


def aggregate(rows):
    result = []
    for sensor, condition in sorted({(r['sensor'], r['condition']) for r in rows}):
        selected = [r for r in rows if (r['sensor'], r['condition']) == (sensor, condition)]
        for key in selected[0]['strata']:
            stats = [r['strata'][key] for r in selected]
            available = [s for s in stats if s['supported']]
            represented = [s for s in stats if s['reference_points']]
            result.append(dict(sensor=sensor, condition=condition, stratum=key,
                               planned_frames=len(selected), frames_with_depth=len(available),
                               reference_points=sum(s['reference_points'] for s in stats),
                               supported=sum(s['supported'] for s in stats),
                               equal_reference_frame_fraction_within_px={
                                   str(t): float(np.mean([s['fraction_reference_within_px'][str(t)]
                                                         for s in represented]))
                                   if represented else None for t in (1., 3.)},
                               median_of_conditional_frame_medians_px=float(np.median([
                                   s['conditional_median_abs_px'] for s in available]))
                               if available else None,
                               median_of_conditional_frame_signed_medians_px=float(np.median([
                                   s['conditional_median_signed_px'] for s in available]))
                               if available else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError('frozen RGB stereo report changed')
    original = json.loads(SOURCE.read_text())
    hashes = original['input_and_source_sha256'].copy()
    config_path = Path('configs/experiment/registration_ms2_image_cpu.yaml')
    config = yaml.safe_load(config_path.read_text())
    plan = json.loads(Path(config['plan']).read_text())
    root = Path(config['sync_root'])/'sync_data'/plan['sequence']
    depth_root = Path(config['depth_root'])/'proj_depth'/plan['sequence']
    for p in (SOURCE, Path(__file__), Path('scripts/analyze_ms2_cross_stereo.py'),
              Path('scripts/probe_ms2_cross_stereo.py'),
              Path('src/aero_ir/registration/temporal_stereo.py')):
        hashes[str(p)] = file_sha256(p)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f'changed dependency: {p}')
    # Every image, timestamp, depth and calibration read below must already be
    # covered by the frozen source inventory; adding an unknown input is forbidden.
    def checked(path):
        if str(path) not in hashes or file_sha256(path) != hashes[str(path)]:
            raise ValueError(f'unrecorded or changed input: {path}')
        return path
    calib = read_calibration(checked(root/'calib.npy').read_bytes())
    times = {(s, side): parse_timestamps(checked(root/s/f'img_{side}_timestamp.txt').read_bytes(),
                                         expected_count=plan['frame_count'])
             for s in ('rgb', 'thr') for side in ('left', 'right')}
    poses = read_rgb_poses(checked(Path(config['odom'])), plan)
    fixed = dict(rgb=np.eye(4), thr=via_common_camera(
        from_millimetres(calib['R_nir2thr'], calib['T_nir2thr']),
        from_millimetres(calib['R_nir2rgb'], calib['T_nir2rgb'])))
    focal = float(calib['K_thrL'][0, 0])
    conditions = ('paired_static', 'paired_timed', 'unrelated_right_timed')
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(schema='ms2_lidar_stereo_plan_v1', input_and_source_sha256=hashes,
                     frame_ids=plan['frame_ids'], sensors=['rgb', 'thr'], conditions=conditions,
                     thermal_window_dn=[3308., 4974.], all_nonzero_depth_reference=True,
                     wrong_pair_cyclic_offset=8, maximum_extrapolation_ns=0,
                     equivalent_disparity_focal_px=focal, opencv_version=cv2.__version__,
                     lidar_source='single-scan depth, not depth_multi or depth_filtered',
                     unavailable_cases_remain_in_denominator=True,
                     camera_fit=False, exploratory=True, registration_qualified=False)
    (args.out_dir/'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
    rows, common = [], []
    for sensor in ('rgb', 'thr'):
        images = {}
        for frame in plan['frame_ids']:
            for side in ('left', 'right'):
                raw = read(checked(root/sensor/f'img_{side}'/f'{frame}.png'))
                images[frame, side] = (cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY) if sensor == 'rgb'
                                        else display_thermal(raw, (3308., 4974.)))
        baseline = from_millimetres(calib[f'R_{sensor}R'], calib[f'T_{sensor}R'])
        kl, kr = calib[f'K_{sensor}L'], calib[f'K_{sensor}R']
        for index, frame in enumerate(plan['frame_ids']):
            depth_map = ms2_depth_metres(read(checked(depth_root/sensor/'depth'/f'{frame}.png')))
            yy, xx = np.nonzero(np.isfinite(depth_map))
            xy, depth = np.c_[xx, yy].astype(float), depth_map[yy, xx]
            if depth_map.shape != images[frame, 'left'].shape:
                raise ValueError('image and depth dimensions differ')
            timed, pose_info = timed_relative(poses, times['rgb', 'left'],
                                              int(times[sensor, 'left'][int(frame)]),
                                              int(times[sensor, 'right'][int(frame)]),
                                              fixed[sensor], baseline)
            wrong = plan['frame_ids'][(index+8) % len(plan['frame_ids'])]
            comparisons = {}
            for condition, relative, right_frame in (
                ('paired_static', baseline, frame), ('paired_timed', timed, frame),
                ('unrelated_right_timed', timed, wrong),
            ):
                estimated = np.full(len(depth), np.nan)
                errors = np.full(len(depth), np.nan)
                native_errors = np.full(len(depth), np.nan)
                geometry, stereo = {}, {}
                if relative is not None:
                    geometry = rectification(kl, kr, relative, depth_map.shape)
                    stereo = estimate_pair(images[frame, 'left'], images[right_frame, 'right'],
                                           geometry)
                    estimated, supported = sample_native_depth(stereo, geometry, xy)
                    predicted = expected_disparity(xy, depth, geometry)
                    observed, _ = sample_disparity(stereo, native_to_rectified(xy, geometry))
                    native_errors = (observed-predicted)*kl[0, 0]/geometry['p1'][0, 0]
                    native_errors[~supported] = np.nan
                    errors = native_errors*focal/kl[0, 0]
                depth_scores, _, _ = depth_statistics(depth, estimated, np.ones(len(depth), bool))
                path = args.out_dir/f'{sensor}_{frame}_{condition}.npz'
                np.savez_compressed(path, **stereo,
                                    **{f'geometry_{k}': v for k, v in geometry.items()},
                                    source_xy=xy, lidar_depth_m=depth, estimated_depth_m=estimated,
                                    native_disparity_errors=native_errors,
                                    equivalent_disparity_errors=errors)
                rows.append(dict(sensor=sensor, frame=frame, condition=condition,
                                 right_frame=right_frame, geometry_available=relative is not None,
                                 timestamp_skew_ns=int(times[sensor, 'right'][int(frame)]
                                                       - times[sensor, 'left'][int(frame)]),
                                 timed_pose_info=pose_info, artifact=path.name,
                                 sha256=file_sha256(path), depth_scores=depth_scores,
                                 strata=summarize_by_depth(errors, depth),
                                 native_pixel_scores=summarize_errors(
                                     native_errors, np.ones(len(depth), bool))))
                comparisons[condition] = errors
            shared = np.isfinite(comparisons['paired_static']) & np.isfinite(
                comparisons['paired_timed'])
            common.append(dict(sensor=sensor, frame=frame, common_points=int(shared.sum()),
                               scores={c: summarize_errors(comparisons[c], shared)
                                       for c in ('paired_static', 'paired_timed')}))
            print(f'{sensor} {frame}: LiDAR reference comparison; timed={timed is not None}',
                  flush=True)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f'dependency changed during experiment: {p}')
    result = dict(schema='ms2_lidar_stereo_v1', rows=rows, aggregate=aggregate(rows),
                  common_support=common, input_and_source_sha256=hashes,
                  preflight_sha256=file_sha256(args.out_dir/'preflight.json'),
                  registration_qualified=False, generator_training_approved=False,
                  note='Sparse projected LiDAR shares calibration and has rasterization, '
                  'occlusion and timing uncertainty. All nonzero reference pixels retained. '
                  'No dense RGB-IR ground truth, physical camera fit or qualification.')
    with (args.out_dir/'report.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out_dir / "report.json"}; not registration qualification')


if __name__ == '__main__':
    main()
