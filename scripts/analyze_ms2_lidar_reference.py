"""Depth-encoding sensitivity and local sparse-depth strata, not a GT error model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import convolve, maximum_filter, minimum_filter

from aero_ir.registration.calibrated import ms2_depth_metres
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.registration.temporal_stereo import native_to_rectified, sample_native_depth
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_confirmation import compare_values
from scripts.analyze_ms2_cross_stereo import expected_disparity, summarize_errors
from scripts.probe_ms2_lidar_stereo import aggregate, summarize_by_depth
from scripts.probe_ms2_stereo_depth import read

BASE = Path('experiments/ms2_lidar_stereo_01/report.json')
BASE_SHA = 'a53e36e8b2fe7bfa91a9cc349d8e1b2c16d6415e1d434eee2ef6820536316282'
DEPTH_STEP_M = 1 / 256


def reference_strata(depth_map, coefficient, radius):
    """Partition every measured pixel using only sparse LiDAR, never stereo errors.

    Range <=1 equivalent disparity pixel is a descriptive bin, not certification
    of a smooth physical surface. Sparse neighbours can miss an occlusion edge.
    """
    if radius not in (2, 4) or not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('fixed radius 2/4 and positive disparity coefficient required')
    if depth_map.ndim != 2 or np.any(np.isfinite(depth_map) & (depth_map <= 0)):
        raise ValueError('positive finite depth or NaN required')
    valid = np.isfinite(depth_map)
    disparity = np.full(depth_map.shape, np.nan)
    disparity[valid] = coefficient / depth_map[valid]
    size = 2 * radius + 1
    count = convolve(valid.astype(np.int32), np.ones((size, size), np.int32),
                     mode='constant', cval=0)
    high = maximum_filter(np.where(valid, disparity, -np.inf), size=size,
                          mode='constant', cval=-np.inf)
    low = minimum_filter(np.where(valid, disparity, np.inf), size=size,
                         mode='constant', cval=np.inf)
    variation = high[valid] - low[valid]
    enough = count[valid] >= 3  # Includes the centre; at least two other returns.
    return dict(all=np.ones(int(valid.sum()), bool),
                sparse_neighbourhood=count[valid] < 3,
                local_range_le_1px=enough & (variation <= 1),
                local_range_gt_1px=enough & (variation > 1))


def encoding_sensitivity(xy, depth, geometry, focal):
    """Maximum endpoint change for +/- one depth DN, holding all geometry fixed.

    This is NOT a total uncertainty bound: no sensor, rasterization, motion or
    calibration error is included. A nonpositive lower endpoint stays unavailable.
    """
    nominal = expected_disparity(xy, depth, geometry)
    lower = expected_disparity(xy, depth - DEPTH_STEP_M, geometry)
    upper = expected_disparity(xy, depth + DEPTH_STEP_M, geometry)
    return np.maximum(np.abs(lower - nominal), np.abs(upper - nominal)) * (
        focal / geometry['p1'][0, 0])


def sensitivity_summary(errors, bound):
    known = np.isfinite(bound)
    measured = np.isfinite(errors)
    within = measured & known & (np.abs(errors) <= bound)
    return dict(reference_points=len(bound), finite_sensitivity_points=int(known.sum()),
                supported=int(measured.sum()),
                supported_with_finite_sensitivity=int((measured & known).sum()),
                supported_residual_within_one_dn_change=int(within.sum()),
                median_change_px=float(np.median(bound[known])) if known.any() else None,
                p95_change_px=float(np.percentile(bound[known], 95)) if known.any() else None,
                max_change_px=float(np.max(bound[known])) if known.any() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(BASE) != BASE_SHA:
        raise ValueError('frozen baseline report changed')
    report = json.loads(BASE.read_text())
    pre_path = BASE.parent / 'preflight.json'
    if file_sha256(pre_path) != report['preflight_sha256']:
        raise ValueError('baseline preflight changed')
    pre = json.loads(pre_path.read_text())
    hashes = report['input_and_source_sha256'].copy()
    config_path = Path('configs/experiment/registration_ms2_image_cpu.yaml')
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'changed input: {path}')
    config = yaml.safe_load(config_path.read_text())
    plan = json.loads(Path(config['plan']).read_text())
    expected = {(s, f, c) for s in pre['sensors'] for f in plan['frame_ids']
                for c in pre['conditions']}
    keys = [(r['sensor'], r['frame'], r['condition']) for r in report['rows']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('missing or duplicate baseline case')
    for path in (BASE, pre_path, Path(__file__)):
        hashes[str(path)] = file_sha256(path)
    focal = pre['equivalent_disparity_focal_px']
    rows, sensitivity, partitions = [], [], {}
    depth_root = Path(config['depth_root']) / 'proj_depth' / plan['sequence']
    for row in report['rows']:
        artifact = BASE.parent / row['artifact']
        if artifact.resolve().parent != BASE.parent.resolve():
            raise ValueError('artifact escapes baseline directory')
        if file_sha256(artifact) != row['sha256']:
            raise ValueError('changed baseline artifact')
        hashes[str(artifact)] = row['sha256']
        with np.load(artifact, allow_pickle=False) as saved:
            data = dict(saved)
        raw_path = depth_root / row['sensor'] / 'depth' / (row['frame'] + '.png')
        if str(raw_path) not in hashes or file_sha256(raw_path) != hashes[str(raw_path)]:
            raise ValueError('unrecorded depth input')
        depth_map = ms2_depth_metres(read(raw_path))
        yy, xx = np.nonzero(np.isfinite(depth_map))
        xy, depth = np.c_[xx, yy].astype(float), depth_map[yy, xx]
        np.testing.assert_array_equal(xy, data['source_xy'])
        np.testing.assert_array_equal(depth, data['lidar_depth_m'])
        geometry = {k.removeprefix('geometry_'): v for k, v in data.items()
                    if k.startswith('geometry_')}
        errors = np.full(len(depth), np.nan)
        estimated = np.full(len(depth), np.nan)
        bound = np.full(len(depth), np.nan)
        if geometry:
            estimated, supported = sample_native_depth(data, geometry, xy)
            observed, _ = sample_disparity(data, native_to_rectified(xy, geometry))
            errors = (observed - expected_disparity(xy, depth, geometry)) * (
                focal / geometry['p1'][0, 0])
            errors[~supported] = np.nan
            bound = encoding_sensitivity(xy, depth, geometry, focal)
        np.testing.assert_allclose(errors, data['equivalent_disparity_errors'],
                                   atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_allclose(estimated, data['estimated_depth_m'],
                                   atol=1e-12, rtol=0, equal_nan=True)
        compare_values(summarize_by_depth(errors, depth), row['strata'])
        key = row['sensor'], row['frame']
        if row['condition'] == 'paired_static':
            coefficient = focal * abs(geometry['p2'][0, 3] / geometry['p1'][0, 0])
            partitions[key] = {r: reference_strata(depth_map, coefficient, r) for r in (2, 4)}
        if key not in partitions:
            raise ValueError('static reference must precede other cases in frozen report')
        for radius, masks in partitions[key].items():
            rows.append(dict(sensor=row['sensor'], frame=row['frame'],
                             condition=row['condition'], radius=radius,
                             strata={k: summarize_errors(errors, m) for k, m in masks.items()}))
        sensitivity.append(dict(sensor=row['sensor'], frame=row['frame'],
                                condition=row['condition'], **sensitivity_summary(errors, bound)))
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'input changed during analysis: {path}')
    result = dict(schema='ms2_lidar_reference_sensitivity_v1', rows=rows,
                  aggregate={str(r): aggregate([row for row in rows if row['radius'] == r])
                             for r in (2, 4)}, encoding_sensitivity=sensitivity,
                  verified_artifact_sha256=hashes, depth_perturbation_m=DEPTH_STEP_M,
                  radii_native_px=[2, 4], minimum_neighbours_including_centre=3,
                  local_range_split_equivalent_px=1., exploratory=True,
                  registration_qualified=False, generator_training_approved=False,
                  note='All reference pixels retained. Local sparse-depth range is not an '
                  'occlusion label. One-DN sensitivity is not total reference uncertainty. '
                  'No calibration fit, error-based filtering or qualification threshold change.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out}; not registration qualification')


if __name__ == '__main__':
    main()
