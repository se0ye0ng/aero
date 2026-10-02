"""CPU-only comparison with reconstructed validity masks; never a qualification gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.raft_stereo import (
    COMMIT,
    ITERATIONS,
    WEIGHTS_SHA,
    consistency,
    rectify_images,
)
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_confirmation import compare_values
from scripts.analyze_ms2_cross_stereo import strata, summarize_errors
from scripts.probe_ms2_raft_stereo import BASE, dependencies, score, verify

LABELS = ('sensor', 'frame', 'condition', 'right_frame', 'geometry_available')


def validate_protocol(baseline, result, preflight, focal):
    if (result.get('schema') != 'ms2_raft_stereo_v1'
            or preflight.get('schema') != 'ms2_raft_stereo_plan_v1'):
        raise ValueError('unexpected result/preflight schema')
    expected = dict(vendor_commit=COMMIT, checkpoint_sha256=WEIGHTS_SHA, iterations=ITERATIONS,
                    precision='float32', correlation='reg', image_resize=False, padding_divisor=32,
                    equivalent_disparity_focal_px=focal, positive_disparity_range=[0., 128.],
                    lr_tolerance_px=1., device='cuda', smoke_only=False, no_finetuning=True,
                    registration_qualified=False, generator_training_approved=False,
                    input='native RGB color; thermal original window replicated to 3 channels',
                    representation_matches_sgbm=False,
                    support='same 5x5 remapping boundary and four-neighbor sampling rules')
    for key, value in expected.items():
        actual = preflight.get(key)
        if actual != value or (isinstance(value, bool) and type(actual) is not bool):
            raise ValueError(f'changed inference protocol: {key}')
    if not isinstance(preflight.get('gpu'), str) or not preflight['gpu'].strip():
        raise ValueError('missing recorded GPU identity')
    if any(result.get(k) is not False
           for k in ('registration_qualified', 'generator_training_approved')):
        raise ValueError('unjustified qualification claim')
    cases = [{k: r[k] for k in LABELS} for r in baseline['rows']]
    if preflight.get('cases') != cases:
        raise ValueError('preflight cases differ from frozen baseline')
    wanted = {(r['sensor'], r['frame'], r['condition']): r for r in cases}
    rows = result.get('rows', [])
    keys = [(r['sensor'], r['frame'], r['condition']) for r in rows]
    if len(keys) != len(wanted) or set(keys) != set(wanted):
        raise ValueError('missing or duplicate inference cases')
    for row, key in zip(rows, keys, strict=True):
        if any(row.get(k) != wanted[key][k] for k in LABELS):
            raise ValueError(f'case label/pairing changed: {key}')
        if type(row['geometry_available']) is not bool:
            raise ValueError('geometry availability must be boolean')


def validate_support(arrays, geometry):
    if not geometry:
        if set(arrays) != {'estimated_depth_m', 'equivalent_disparity_errors'}:
            raise ValueError('unavailable pose case contains unexpected stereo arrays')
        return
    shape = tuple(int(v) for v in geometry['shape'])
    for key in ('disparity', 'reverse_disparity'):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'invalid raw model disparity: {key}')
    # Remapping support depends on geometry, not image intensities. Using blank
    # images here avoids another learned-model execution and leaves the mask exact.
    blank = np.zeros(shape, np.uint8)
    _, supports = rectify_images(blank, blank, geometry)
    rebuilt = consistency(arrays['disparity'], arrays['reverse_disparity'], *supports)
    for key in ('valid', 'rectified_left_support', 'rectified_right_support'):
        if arrays[key].dtype != bool or not np.array_equal(arrays[key], rebuilt[key]):
            raise ValueError(f'saved support does not follow frozen rules: {key}')
    np.testing.assert_allclose(arrays['lr_error'], rebuilt['lr_error'], atol=1e-12, rtol=0,
                               equal_nan=True)


def paired_statistics(baseline_error, candidate_error, depth):
    if (baseline_error.shape != depth.shape or candidate_error.shape != depth.shape
            or depth.ndim != 1 or not np.isfinite(depth).all() or (depth <= 0).any()):
        raise ValueError('aligned errors and positive measured depth required')
    result = {}
    for key, mask in strata(depth).items():
        common = mask & np.isfinite(baseline_error) & np.isfinite(candidate_error)
        result[key.replace('rgb_depth', 'lidar_depth')] = dict(
            baseline=summarize_errors(baseline_error, mask),
            raft=summarize_errors(candidate_error, mask),
            common=dict(baseline=summarize_errors(baseline_error, common),
                        raft=summarize_errors(candidate_error, common)))
    return result


def aggregate_comparison(rows):
    result = []
    for sensor, condition in sorted({(r['sensor'], r['condition']) for r in rows}):
        selected = [r for r in rows if (r['sensor'], r['condition']) == (sensor, condition)]
        for key in selected[0]['strata']:
            stats = [r['strata'][key] for r in selected]
            represented = [r for r in stats if r['baseline']['reference_points']]
            shared = [r['common'] for r in stats if r['common']['baseline']['reference_points']]
            record = dict(sensor=sensor, condition=condition, stratum=key,
                          planned_frames=len(selected), reference_frames=len(represented),
                          reference_points=sum(r['baseline']['reference_points'] for r in stats),
                          common_frames=len(shared),
                          common_points=sum(r['baseline']['reference_points'] for r in shared))
            for model in ('baseline', 'raft'):
                available = [r[model] for r in stats if r[model]['supported']]
                record[model] = dict(
                    supported=sum(r[model]['supported'] for r in stats),
                    frames_with_estimate=len(available),
                    equal_frame_full_fraction_within_px={str(t): float(np.mean([
                        r[model]['fraction_reference_within_px'][str(t)] for r in represented]))
                        if represented else None for t in (1., 3.)},
                    conditional_median_of_frame_medians_px=float(np.median([
                        r['conditional_median_abs_px'] for r in available])) if available else None,
                    common_median_of_frame_medians_px=(float(np.median([
                        r[model]['conditional_median_abs_px'] for r in shared]))
                        if shared else None))
            deltas = [r['raft']['conditional_median_abs_px']
                      - r['baseline']['conditional_median_abs_px'] for r in shared]
            record['common_frame_medians'] = dict(
                improved=int(np.count_nonzero(np.asarray(deltas) < -1e-12)),
                worsened=int(np.count_nonzero(np.asarray(deltas) > 1e-12)),
                tied=int(np.count_nonzero(np.abs(deltas) <= 1e-12)))
            result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('experiments/ms2_raft_stereo_01'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    path = args.run_dir/'report.json'
    if not path.is_file():
        raise FileNotFoundError(f'GPU report not found: {path}; run scripts/run_ms2_raft_stereo.sh')
    if args.out.exists():
        raise FileExistsError(args.out)
    baseline, hashes = dependencies()
    result = json.loads(path.read_text())
    pre_path = args.run_dir/'preflight.json'
    pre = json.loads(pre_path.read_text())
    base_pre = json.loads((BASE.parent/'preflight.json').read_text())
    focal = base_pre['equivalent_disparity_focal_px']
    validate_protocol(baseline, result, pre, focal)
    if pre.get('input_and_source_sha256') != hashes:
        raise ValueError('preflight provenance differs from frozen inputs')
    for p in (path, pre_path, Path(__file__)):
        hashes[str(p)] = file_sha256(p)
    verify(args.run_dir)
    original = {(r['sensor'], r['frame'], r['condition']): r for r in baseline['rows']}
    rows = []
    for row in result['rows']:
        ref = original[row['sensor'], row['frame'], row['condition']]
        with np.load(BASE.parent/ref['artifact'], allow_pickle=False) as data:
            saved = dict(data)
        artifact = args.run_dir/row['artifact']
        hashes[str(artifact)] = row['sha256']
        with np.load(artifact, allow_pickle=False) as data:
            arrays = dict(data)
        geometry = {k.removeprefix('geometry_'): v for k, v in saved.items()
                    if k.startswith('geometry_')}
        validate_support(arrays, geometry)
        # Reconstruct the older estimator too, not only the new model's scores.
        z, errors, scores = score(saved, saved, focal)
        np.testing.assert_allclose(z, saved['estimated_depth_m'], atol=1e-12, rtol=0,
                                   equal_nan=True)
        np.testing.assert_allclose(errors, saved['equivalent_disparity_errors'], atol=1e-12,
                                   rtol=0, equal_nan=True)
        compare_values(ref['strata'], scores['strata'])
        compare_values(ref['depth_scores'], scores['depth_scores'])
        rows.append({k: row[k] for k in LABELS} | dict(strata=paired_statistics(
            errors, arrays['equivalent_disparity_errors'], saved['lidar_depth_m'])))
    control = args.run_dir/result['synthetic_control']['artifact']
    hashes[str(control)] = result['synthetic_control']['sha256']
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f'input changed during analysis: {p}')
    analysis = dict(schema='ms2_raft_stereo_comparison_v1', rows=rows,
                    comparison=aggregate_comparison(rows), verified_artifact_sha256=hashes,
                    registration_qualified=False, generator_training_approved=False,
                    note='Descriptive comparison on the same LiDAR reference population. '
                    'Coverage and common-point accuracy are distinct. Not independent '
                    'dense RGB-IR correspondence GT or a new qualification threshold.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(analysis, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out}; no registration qualification or generator approval')


if __name__ == '__main__':
    main()
