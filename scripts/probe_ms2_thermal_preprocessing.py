"""Exploratory radiometric control; frozen author geometry, no qualification claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.calibrated import pixel_support
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.registration.temporal_stereo import (
    estimate_pair,
    native_to_rectified,
    sample_native_depth,
)
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_cross_stereo import aggregate, expected_disparity, strata, summarize_errors
from scripts.probe_ms2_cross_stereo import depth_statistics

MODES = ('original_fixed', 'pair_shared_percentiles', 'independent_percentiles')
REPORT = Path('experiments/ms2_cross_stereo_02/report.json')
REPORT_SHA = 'c237aa8b490943a222276f52cb662d5c0ab646dfcecec34bafa8d4250625d794'
PLAN = Path('experiments/ms2_intrinsic_confirmation_plan_01/plan.json')
PLAN_SHA = '68dbed6ba590eef37b60d41f0c32d48971a18535dfbd8b7bfa0023556d467ce3'


def preprocess_pair(left, right, mode, fixed_window=(3308., 4974.)):
    """Per-image percentiles use full images, never matches, depth or output scores.

    Shared bounds are min(p1_left,p1_right), max(p99_left,p99_right).
    Independent bounds remove some relative intensity differences but may also
    distort real differences from distinct fields of view. Neither calibrates DN.
    """
    if mode not in MODES:
        raise ValueError('unknown preprocessing mode')
    if any(x.dtype != np.uint16 or x.ndim != 2 or not x.size for x in (left, right)):
        raise ValueError('nonempty uint16 thermal images required')
    if left.shape != right.shape:
        raise ValueError('equal image shapes required')
    fixed_window = np.asarray(fixed_window, float)
    if (fixed_window.shape != (2,) or not np.isfinite(fixed_window).all()
            or fixed_window[1] <= fixed_window[0]):
        raise ValueError('increasing finite fixed window required')
    quantiles = [np.percentile(x, [1., 50., 99.]) for x in (left, right)]
    windows = [np.asarray(fixed_window, float)] * 2
    if mode == 'pair_shared_percentiles':
        windows = [np.array([min(q[0] for q in quantiles),
                             max(q[2] for q in quantiles)])] * 2
    elif mode == 'independent_percentiles':
        windows = [q[[0, 2]] for q in quantiles]
    images, audit = [], []
    for raw, q, window in zip((left, right), quantiles, windows, strict=True):
        low, high = window
        # Constant percentile range contains no contrast: retain a blank image,
        # not a deleted frame or an invented window selected using the score.
        flat = bool(high == low)
        image = np.zeros(raw.shape, np.uint8) if flat else display_thermal(raw, window)
        images.append(image)
        audit.append(dict(percentiles_1_50_99=q.tolist(), window_dn=window.tolist(),
                          flat_percentile_range=flat,
                          clipped_fraction=float(np.mean((raw < low) | (raw > high)))))
    return images, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(REPORT) != REPORT_SHA or file_sha256(PLAN) != PLAN_SHA:
        raise ValueError('frozen source report or plan changed')
    previous, plan = json.loads(REPORT.read_text()), json.loads(PLAN.read_text())
    hashes = previous['input_and_source_sha256'].copy()
    source_rows = [r for r in previous['rows'] if r['camera'] == 'author_calibration'
                   and r['condition'] != 'paired_static']
    expected = {(f, c) for f in plan['frame_ids']
                for c in ('paired_timed', 'unrelated_right_timed')}
    if (len(source_rows) != len(expected)
            or {(r['frame'], r['condition']) for r in source_rows} != expected):
        raise ValueError('incomplete source cases')
    for row in source_rows:
        path = REPORT.parent/row['artifact']
        if path.resolve().parent != REPORT.parent.resolve():
            raise ValueError('unsafe source artifact')
        hashes[str(path)] = row['sha256']
    for path in (REPORT, PLAN, Path(__file__), Path('scripts/analyze_ms2_cross_stereo.py')):
        hashes[str(path)] = file_sha256(path)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'changed dependency: {path}')
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(schema='ms2_thermal_preprocessing_plan_v1', modes=MODES,
                     source_cases=source_rows, input_and_source_sha256=hashes,
                     percentile_bounds=[1., 99.], fixed_window=plan['thermal_window_dn'],
                     camera='author_calibration', stereo_parameters_changed=False,
                     same_reference_denominator=True, exploration_on_observed_panel=True,
                     opencv_version=cv2.__version__, numpy_version=np.__version__,
                     refit=False, registration_qualified=False, generator_training_approved=False)
    (args.out_dir/'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
    root = Path('experiments/ms2_intrinsic_confirmation_sync_01/sync_data')/plan['sequence']
    rows, common = [], []
    for row in source_rows:
        with np.load(REPORT.parent/row['artifact'], allow_pickle=False) as file:
            saved = dict(file)
        geometry = {k.removeprefix('geometry_'): v for k, v in saved.items()
                    if k.startswith('geometry_')}
        raw = []
        for side, frame in (('left', row['frame']), ('right', row['right_frame'])):
            path = root/'thr'/f'img_{side}'/f'{frame}.png'
            if str(path) not in hashes:
                raise ValueError('unrecorded image input')
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape != (256, 640) or image.dtype != np.uint16:
                raise ValueError('invalid thermal image')
            raw.append(image)
        reference = saved['reference_supported']
        xy, z = saved['projected_xy'], saved['expected_z']
        eligible = reference & pixel_support(xy, (256, 640)) & np.isfinite(z) & (z > 0)
        predicted = expected_disparity(xy[eligible], z[eligible], geometry)
        rectified = native_to_rectified(xy[eligible], geometry)
        errors_by_mode = {}
        for mode in MODES:
            images, audit = preprocess_pair(*raw, mode, plan['thermal_window_dn'])
            stereo = estimate_pair(*images, geometry)
            estimated = np.full(len(z), np.nan)
            estimated[eligible], _ = sample_native_depth(stereo, geometry, xy[eligible])
            depth_scores, _, supported = depth_statistics(z, estimated, reference)
            if mode == 'original_fixed':
                for key in stereo:
                    if not np.array_equal(stereo[key], saved[key], equal_nan=True):
                        raise ValueError(f'baseline stereo does not reproduce: {key}')
                if not np.allclose(estimated, saved['estimated_z'], rtol=0, atol=1e-12,
                                   equal_nan=True):
                    raise ValueError('baseline depth does not reproduce')
            observed, _ = sample_disparity(stereo, rectified)
            errors = np.full(len(z), np.nan)
            errors[eligible] = ((observed-predicted)*geometry['left_k'][0, 0]
                                / geometry['p1'][0, 0])
            errors[~supported] = np.nan
            errors_by_mode[mode] = errors
            artifact = args.out_dir/f"{row['frame']}_{row['condition']}_{mode}.npz"
            np.savez_compressed(artifact, **stereo, estimated_z=estimated,
                                equivalent_disparity_errors=errors)
            rows.append(dict(frame=row['frame'], condition=row['condition'], camera=mode,
                             source_artifact=row['artifact'], right_frame=row['right_frame'],
                             artifact=artifact.name, sha256=file_sha256(artifact),
                             preprocessing_audit=audit, depth_scores=depth_scores,
                             strata={k: summarize_errors(errors, reference & mask)
                                     for k, mask in strata(saved['source_depth_m']).items()}))
        shared = reference & np.logical_and.reduce([
            np.isfinite(e) for e in errors_by_mode.values()])
        common.append(dict(frame=row['frame'], condition=row['condition'],
                           common_points=int(shared.sum()),
                           modes={k: summarize_errors(v, shared)
                                  for k, v in errors_by_mode.items()}))
        print(f"{row['frame']} {row['condition']}: all three preprocessing modes done", flush=True)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'dependency changed during experiment: {path}')
    result = dict(schema='ms2_thermal_preprocessing_v1', rows=rows, aggregate=aggregate(rows),
                  common_support=common, input_and_source_sha256=hashes,
                  preflight_sha256=file_sha256(args.out_dir/'preflight.json'),
                  original_stereo_reproduced=True, registration_qualified=False,
                  generator_training_approved=False,
                  note='Exploratory preprocessing sensitivity on an already observed panel. '
                  'Stereo disparity disagreement is not RGB-IR correspondence error or GT. '
                  'No camera fit, frame exclusion, parameter selection or qualification.')
    with (args.out_dir/'report.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out_dir / "report.json"}; not qualification')


if __name__ == '__main__':
    main()
