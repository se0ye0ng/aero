"""Reconstruct cross-stereo scores and express depth discrepancy in disparity units.

Stereo disparity error is NOT RGB-to-thermal correspondence pixel error. In
particular, two wrongly associated points on an equal-depth surface may agree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.calibrated import pixel_support
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.registration.temporal_stereo import native_to_rectified, sample_native_depth
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_confirmation import compare_values
from scripts.probe_ms2_cross_stereo import depth_statistics


def expected_disparity(native_xy, native_z, geometry):
    """Predict disparity in the per-exposure virtual rectified camera."""
    points, depth = np.asarray(native_xy), np.asarray(native_z)
    if points.shape != (len(depth), 2) or depth.ndim != 1:
        raise ValueError('Nx2 native points and N native Z-depths required')
    rays = np.c_[points, np.ones(len(points))] @ np.linalg.inv(geometry['left_k']).T
    camera = rays * depth[:, None]
    rect_z = (camera @ geometry['r1'].T)[:, 2]
    disparity = np.full(len(points), np.nan)
    valid = np.isfinite(rect_z) & (rect_z > 0) & np.isfinite(depth) & (depth > 0)
    delta_cx = geometry['p1'][0, 2]-geometry['p2'][0, 2]
    disparity[valid] = delta_cx-geometry['p2'][0, 3]/rect_z[valid]
    return disparity


def summarize_errors(signed_errors, mask):
    finite = mask & np.isfinite(signed_errors)
    count, supported = int(mask.sum()), int(finite.sum())
    absolute = np.abs(signed_errors[finite])
    return dict(reference_points=count, supported=supported,
                fraction_reference_within_px={str(t): float(np.count_nonzero(
                    finite & (np.abs(signed_errors) <= t))/count) if count else None
                                              for t in (1., 3.)},
                fraction_available_within_px={str(t): float(np.mean(absolute <= t))
                                             if supported else None for t in (1., 3.)},
                conditional_median_abs_px=float(np.median(absolute)) if supported else None,
                conditional_p95_abs_px=float(np.percentile(absolute, 95)) if supported else None,
                conditional_median_signed_px=float(np.median(signed_errors[finite]))
                if supported else None)


def strata(depth):
    result = {'all': np.ones(len(depth), bool)}
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, np.inf)):
        result[f'rgb_depth_m_{lo}_{hi}'] = (depth >= lo) & (depth < hi)
    return result


def aggregate(rows):
    result = []
    for camera, condition in sorted({(r['camera'], r['condition']) for r in rows}):
        selected = [r for r in rows if (r['camera'], r['condition']) == (camera, condition)]
        for key in selected[0]['strata']:
            stats = [r['strata'][key] for r in selected]
            represented = [s for s in stats if s['reference_points']]
            available = [s for s in stats if s['supported']]
            result.append(dict(camera=camera, condition=condition, stratum=key,
                               planned_frames=len(selected), reference_frames=len(represented),
                               frames_with_depth=len(available),
                               reference_points=sum(s['reference_points'] for s in stats),
                               supported=sum(s['supported'] for s in stats),
                               equal_reference_frame_fraction_within_px={
                                   str(t): float(np.mean([s['fraction_reference_within_px'][str(t)]
                                                         for s in represented]))
                                   if represented else None for t in (1., 3.)},
                               equal_available_frame_fraction_within_px={
                                   str(t): float(np.mean([s['fraction_available_within_px'][str(t)]
                                                         for s in available]))
                                   if available else None for t in (1., 3.)},
                               median_of_conditional_frame_medians_px=float(np.median(
                                   [s['conditional_median_abs_px'] for s in available]))
                               if available else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    path = Path('experiments/ms2_cross_stereo_02/report.json')
    if file_sha256(path) != 'c237aa8b490943a222276f52cb662d5c0ab646dfcecec34bafa8d4250625d794':
        raise ValueError('frozen cross-stereo report changed')
    report = json.loads(path.read_text())
    preflight_path = path.parent/'preflight.json'
    if file_sha256(preflight_path) != report['preflight_sha256']:
        raise ValueError('cross-stereo preflight changed')
    preflight = json.loads(preflight_path.read_text())
    hashes = report['input_and_source_sha256'].copy()
    for source, digest in hashes.items():
        if file_sha256(source) != digest:
            raise ValueError(f'changed dependency: {source}')
    plan_path = Path('experiments/ms2_intrinsic_confirmation_plan_01/plan.json')
    plan = json.loads(plan_path.read_text())
    cameras = preflight['cameras']
    expected = {(f, c, condition) for f in plan['frame_ids'] for c in cameras
                for condition in ('paired_static', 'paired_timed', 'unrelated_right_timed')}
    keys = [(r['frame'], r['camera'], r['condition']) for r in report['rows']]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('incomplete or duplicated planned cases')
    fixed_focal = cameras['author_calibration']['target_intrinsic'][0][0]
    yy, xx = np.mgrid[8:384:16, 8:1224:16]
    expected_grid = np.c_[xx.ravel(), yy.ravel()].astype(float)
    rows, retained, reference_masks = [], {}, {}
    for row in report['rows']:
        artifact = path.parent/row['artifact']
        if (artifact.parent != path.parent or artifact.resolve().parent != path.parent.resolve()
                or file_sha256(artifact) != row['sha256']):
            raise ValueError('changed or unsafe stereo artifact')
        hashes[str(artifact)] = row['sha256']
        with np.load(artifact, allow_pickle=False) as values:
            arrays = dict(values)
        if not np.array_equal(arrays['source_xy'], expected_grid):
            raise ValueError('source grid changed')
        geometry = {k.removeprefix('geometry_'): v for k, v in arrays.items()
                    if k.startswith('geometry_')}
        reference = arrays['reference_supported']
        if row['frame'] in reference_masks:
            if not np.array_equal(reference_masks[row['frame']], reference):
                raise ValueError('reference denominator changes between cameras or conditions')
        reference_masks[row['frame']] = reference
        native_xy, expected_z = arrays['projected_xy'], arrays['expected_z']
        eligible = reference & pixel_support(native_xy, (256, 640))
        eligible &= np.isfinite(expected_z) & (expected_z > 0)
        estimated = np.full(len(expected_z), np.nan)
        estimated[eligible], _ = sample_native_depth(arrays, geometry, native_xy[eligible])
        if not np.allclose(estimated, arrays['estimated_z'], equal_nan=True, atol=1e-12, rtol=0):
            raise ValueError('thermal depth resampling differs')
        stats, _, jointly_supported = depth_statistics(expected_z, estimated, reference)
        if not np.array_equal(jointly_supported, arrays['jointly_supported']):
            raise ValueError('joint support differs')
        compare_values(row['scores'], stats)
        predicted = expected_disparity(native_xy[eligible], expected_z[eligible], geometry)
        rectified = native_to_rectified(native_xy[eligible], geometry)
        observed, _ = sample_disparity(arrays, rectified)
        errors = np.full(len(reference), np.nan)
        # Each exposure/candidate has its own virtual focal length. Express all
        # angular disparity discrepancies at the same AUTHOR thermal focal scale.
        errors[eligible] = (observed-predicted)*fixed_focal/geometry['p1'][0, 0]
        errors[~jointly_supported] = np.nan
        rows.append({k: row[k] for k in ('frame', 'camera', 'condition')}
                    | dict(strata={key: summarize_errors(errors, mask & reference)
                                   for key, mask in strata(arrays['source_depth_m']).items()}))
        retained[row['frame'], row['camera'], row['condition']] = errors
    common = []
    for frame in plan['frame_ids']:
        errors = {c: retained[frame, c, 'paired_timed'] for c in cameras}
        shared = reference_masks[frame] & np.logical_and.reduce([
            np.isfinite(e) for e in errors.values()])
        common.append(dict(frame=frame, reference_points=int(reference_masks[frame].sum()),
                           common_points=int(shared.sum()),
                           cameras={c: summarize_errors(e, shared) for c, e in errors.items()}))
    for source in (path, preflight_path, plan_path, Path(__file__),
                   Path('scripts/analyze_ms2_confirmation.py')):
        hashes[str(source)] = file_sha256(source)
    for source, digest in hashes.items():
        if file_sha256(source) != digest:
            raise ValueError(f'dependency changed during analysis: {source}')
    result = dict(schema='ms2_cross_stereo_disparity_analysis_v1', rows=rows,
                  aggregate=aggregate(rows), timed_camera_common_support=common,
                  equivalent_focal_px=fixed_focal, verified_artifact_sha256=hashes,
                  registration_qualified=False, generator_training_approved=False,
                  note='Disparity discrepancy at a fixed author focal scale, NOT RGB-to-IR '
                  'correspondence error. Estimates can agree on a wrongly associated equal-depth '
                  'surface. This is score reconstruction, not image-stereo recomputation or GT.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out}; disparity agreement is not registration qualification')


if __name__ == '__main__':
    main()
