"""Frozen RAFT-Stereo versus StereoSGBM on the same sparse LiDAR reference cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.raft_stereo import (
    COMMIT,
    ITERATIONS,
    WEIGHTS_SHA,
    estimate_pair,
    identities,
    infer_disparity,
    load_model,
)
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.registration.temporal_stereo import native_to_rectified, sample_native_depth
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_confirmation import compare_values
from scripts.analyze_ms2_cross_stereo import expected_disparity, summarize_errors
from scripts.probe_ms2_cross_stereo import depth_statistics
from scripts.probe_ms2_lidar_stereo import aggregate, summarize_by_depth
from scripts.probe_ms2_stereo_depth import read

BASE = Path('experiments/ms2_lidar_stereo_01/report.json')
BASE_SHA = 'a53e36e8b2fe7bfa91a9cc349d8e1b2c16d6415e1d434eee2ef6820536316282'
ROOT = Path('experiments/ms2_train_screen_sync_01/sync_data/_2021-08-06-10-59-33')


def dependencies():
    if file_sha256(BASE) != BASE_SHA:
        raise ValueError('frozen LiDAR baseline report changed')
    baseline = json.loads(BASE.read_text())
    hashes = baseline['input_and_source_sha256'] | identities()
    for r in baseline['rows']:
        p = BASE.parent/r['artifact']
        if p.resolve().parent != BASE.parent.resolve():
            raise ValueError('unsafe baseline artifact')
        hashes[str(p)] = r['sha256']
    for p in (BASE, BASE.parent/'preflight.json', Path(__file__),
              Path('src/aero_ir/registration/raft_stereo.py'),
              Path('scripts/probe_ms2_lidar_stereo.py'),
              Path('scripts/run_ms2_raft_stereo.sh'),
              Path('requirements/raft-stereo-runtime.txt')):
        hashes[str(p)] = file_sha256(p)
    if hashes[str(BASE.parent/'preflight.json')] != baseline['preflight_sha256']:
        raise ValueError('baseline preflight changed')
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f'changed dependency: {p}')
    return baseline, hashes


def score(stereo, original, focal):
    xy, depth = original['source_xy'], original['lidar_depth_m']
    estimated, errors = np.full(len(depth), np.nan), np.full(len(depth), np.nan)
    geometry = {k.removeprefix('geometry_'): v for k, v in original.items()
                if k.startswith('geometry_')}
    if geometry:
        estimated, supported = sample_native_depth(stereo, geometry, xy)
        observed, _ = sample_disparity(stereo, native_to_rectified(xy, geometry))
        predicted = expected_disparity(xy, depth, geometry)
        errors = (observed-predicted)*focal/geometry['p1'][0, 0]
        errors[~supported] = np.nan
    reference = np.ones(len(depth), bool)
    depth_scores, _, _ = depth_statistics(depth, estimated, reference)
    common = np.isfinite(errors) & np.isfinite(original['equivalent_disparity_errors'])
    common_scores = dict(baseline=summarize_errors(original['equivalent_disparity_errors'], common),
                         raft=summarize_errors(errors, common))
    return estimated, errors, dict(depth_scores=depth_scores,
                                    strata=summarize_by_depth(errors, depth),
                                    common_support=common_scores)


def synthetic_control(model, padder, out_dir):
    left = np.random.default_rng(0).integers(0, 256, (97, 163), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]
    disparity = infer_disparity(model, padder, left, right)
    mask = np.zeros(left.shape, bool)
    mask[16:-16, 32:-16] = True
    scores = summarize_errors(disparity.ravel()-8., mask.ravel())
    p = out_dir/'synthetic_8px.npz'
    np.savez_compressed(p, left=left, right=right, disparity=disparity, scoring_mask=mask)
    return dict(artifact=p.name, sha256=file_sha256(p), scores=scores,
                ok=scores['fraction_reference_within_px']['1.0'] >= .95,
                note='Known synthetic disparity execution control, not physical RGB-IR evidence')


def verify(out_dir):
    baseline, hashes = dependencies()
    result = json.loads((out_dir/'report.json').read_text())
    if (result['input_and_source_sha256'] != hashes or result['registration_qualified']
            or result['generator_training_approved']
            or file_sha256(out_dir/'preflight.json') != result['preflight_sha256']):
        raise ValueError('changed run provenance or unjustified qualification claim')
    original = {(r['sensor'], r['frame'], r['condition']): r for r in baseline['rows']}
    keys = [(r['sensor'], r['frame'], r['condition']) for r in result['rows']]
    if set(keys) != set(original) or len(keys) != len(original):
        raise ValueError('missing or duplicate cases')
    pre = json.loads((out_dir/'preflight.json').read_text())
    for row in result['rows']:
        ref = original[row['sensor'], row['frame'], row['condition']]
        p = out_dir/row['artifact']
        if p.resolve().parent != out_dir.resolve() or file_sha256(p) != row['sha256']:
            raise ValueError('changed or unsafe model artifact')
        with np.load(BASE.parent/ref['artifact'], allow_pickle=False) as data:
            saved = dict(data)
        with np.load(p, allow_pickle=False) as data:
            arrays = dict(data)
        z, errors, scores = score(arrays, saved, pre['equivalent_disparity_focal_px'])
        np.testing.assert_allclose(z, arrays['estimated_depth_m'], atol=1e-12, rtol=0,
                                   equal_nan=True)
        np.testing.assert_allclose(errors, arrays['equivalent_disparity_errors'],
                                   atol=1e-12, rtol=0, equal_nan=True)
        for key, value in scores.items():
            compare_values(row[key], value)
    if aggregate(result['rows']) != result['aggregate']:
        raise ValueError('aggregate score mismatch')
    control = result['synthetic_control']
    control_path = out_dir/control['artifact']
    if (control_path.resolve().parent != out_dir.resolve()
            or file_sha256(control_path) != control['sha256']):
        raise ValueError('changed synthetic control')
    with np.load(control_path, allow_pickle=False) as data:
        expected_left = np.random.default_rng(0).integers(0, 256, (97, 163), dtype=np.uint8)
        expected_right = np.zeros_like(expected_left)
        expected_right[:, :-8] = expected_left[:, 8:]
        expected_mask = np.zeros(expected_left.shape, bool)
        expected_mask[16:-16, 32:-16] = True
        np.testing.assert_array_equal(data['left'], expected_left)
        np.testing.assert_array_equal(data['right'], expected_right)
        np.testing.assert_array_equal(data['scoring_mask'], expected_mask)
        if data['disparity'].shape != expected_left.shape:
            raise ValueError('synthetic disparity shape mismatch')
        scores = summarize_errors(data['disparity'].ravel()-8., data['scoring_mask'].ravel())
    compare_values(control['scores'], scores)
    if not control['ok'] or scores['fraction_reference_within_px']['1.0'] < .95:
        raise ValueError('failed synthetic execution control')
    print(f'Verified {len(keys)} cases and score reconstruction; not registration qualification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if sum((args.smoke_only, args.verify_only, args.preflight_only)) > 1:
        parser.error('choose only one special execution mode')
    if args.verify_only:
        verify(args.out_dir)
        return
    baseline, hashes = dependencies()
    if args.out_dir.exists():
        raise FileExistsError(f'refusing to overwrite: {args.out_dir}')
    if args.preflight_only:
        print(f'Preflight passed: {len(baseline["rows"])} cases; {len(hashes)} identities.')
        return
    if args.device == 'cpu' and not args.smoke_only:
        parser.error('CPU mode is restricted to the bounded --smoke-only execution check')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    model, padder = load_model(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(schema='ms2_raft_stereo_plan_v1', input_and_source_sha256=hashes,
                     cases=[{k: r[k] for k in ('sensor', 'frame', 'condition', 'right_frame',
                                               'geometry_available')} for r in baseline['rows']],
                     vendor_commit=COMMIT, checkpoint_sha256=WEIGHTS_SHA,
                     iterations=ITERATIONS, precision='float32', correlation='reg',
                     input='native RGB color; thermal original window replicated to 3 channels',
                     representation_matches_sgbm=False, image_resize=False, padding_divisor=32,
                     equivalent_disparity_focal_px=387.78695052062903,
                     positive_disparity_range=[0., 128.], lr_tolerance_px=1.,
                     support='same 5x5 remapping boundary and four-neighbor sampling rules',
                     device=args.device, torch_version=torch.__version__,
                     cuda_version=torch.version.cuda, opencv_version=cv2.__version__,
                     smoke_only=args.smoke_only, no_finetuning=True,
                     registration_qualified=False, generator_training_approved=False)
    if args.device == 'cuda':
        preflight['gpu'] = torch.cuda.get_device_name(0)
    (args.out_dir/'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
    control = synthetic_control(model, padder, args.out_dir)
    (args.out_dir/'control.json').write_text(json.dumps(control, indent=2, allow_nan=False))
    if not control['ok']:
        raise RuntimeError('synthetic disparity execution check failed; no real cases launched')
    if args.smoke_only:
        print(f'CPU/model execution control passed: {args.out_dir / "control.json"}')
        return
    rows = []
    for index, ref in enumerate(baseline['rows']):
        with np.load(BASE.parent/ref['artifact'], allow_pickle=False) as data:
            saved = dict(data)
        geometry = {k.removeprefix('geometry_'): v for k, v in saved.items()
                    if k.startswith('geometry_')}
        stereo = {}
        if ref['geometry_available']:
            images = []
            for side, frame in (('left', ref['frame']), ('right', ref['right_frame'])):
                p = ROOT/ref['sensor']/f'img_{side}'/f'{frame}.png'
                if str(p) not in hashes or file_sha256(p) != hashes[str(p)]:
                    raise ValueError(f'unrecorded or changed image: {p}')
                raw = read(p)
                images.append(raw if ref['sensor'] == 'rgb'
                              else display_thermal(raw, (3308., 4974.)))
            stereo = estimate_pair(model, padder, *images, geometry)
        z, errors, scores = score(stereo, saved, preflight['equivalent_disparity_focal_px'])
        p = args.out_dir/f"{ref['sensor']}_{ref['frame']}_{ref['condition']}.npz"
        np.savez_compressed(p, **stereo, estimated_depth_m=z, equivalent_disparity_errors=errors)
        rows.append({k: ref[k] for k in ('sensor', 'frame', 'condition', 'right_frame',
                                         'geometry_available')}
                    | dict(artifact=p.name, sha256=file_sha256(p), **scores))
        print(f'{index+1}/{len(baseline["rows"])} {p.stem}', flush=True)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f'input/source changed during inference: {p}')
    report = dict(schema='ms2_raft_stereo_v1', rows=rows, aggregate=aggregate(rows),
                  synthetic_control=control, input_and_source_sha256=hashes,
                  preflight_sha256=file_sha256(args.out_dir/'preflight.json'),
                  registration_qualified=False, generator_training_approved=False,
                  note='Exploratory pretrained-estimator comparison, not independent dense GT. '
                  'RGB color versus SGBM grayscale is an explicit representation difference. '
                  'All original LiDAR references and unavailable pose cases retained.')
    with (args.out_dir/'report.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out_dir / "report.json"}; not registration qualification')


if __name__ == '__main__':
    main()
