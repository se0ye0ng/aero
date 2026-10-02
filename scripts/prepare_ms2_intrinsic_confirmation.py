"""Freeze effective-camera candidates before extracting a disjoint train-image panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.data.ms2_archive import extract_screen
from aero_ir.registration.calibrated import intrinsic_matrix, rigid_matrix
from aero_ir.utils.manifest import file_sha256
from scripts.prepare_ms2_confirmation import interval_midpoints

BASE_PATH = Path('experiments/ms2_train_screen_plan_01/plan.json')
BASE_SHA256 = '681cf9c6f3d680cf4d0ee7c3e37575868a56066fa3f1bf19662466dedb2952d0'
CANDIDATE_PATH = Path('experiments/ms2_intrinsic_sensitivity_01/report.json')
CANDIDATE_SHA256 = 'd71ad7849fbc26bc1395b5475ed79978099bd148399a19a039df69c58b40e2ff'
ARCHIVE = Path('experiments/external/ms2_first_train_archives/sync_data.tar.bz2.part')
ARCHIVE_SHA256 = 'c3a10f2cef0d04ea6999c9885311aa1bda206e3c228960ce3e1462d19e913902'
PLAN_PATH = Path('experiments/ms2_intrinsic_confirmation_plan_01/plan.json')


def quarter_frames(frames):
    midpoints = interval_midpoints(frames)
    values = [f'{(int(a)+int(m))//2:06d}' for a, m in zip(frames[:-1], midpoints, strict=True)]
    if len(set(values)) != len(values) or set(values) & (set(frames) | set(midpoints)):
        raise ValueError('quarter panel overlaps prior observed frames')
    return values


def make_plan(base, candidates):
    expected = {'author_calibration', 'xoftr_640_rotation_only', 'minima_xoftr_rotation_only',
                'xoftr_640_rotation_plus_intrinsic', 'minima_xoftr_rotation_plus_intrinsic'}
    names = [c['name'] for c in candidates['candidates']]
    if set(names) != expected or len(names) != len(expected):
        raise ValueError('unexpected or duplicate frozen camera candidates')
    cameras = {}
    for candidate in candidates['candidates']:
        if 'parameters' in candidate and (not candidate['optimizer_success']
                                           or any(candidate['bound_hit'])):
            raise ValueError('unconverged or boundary candidate cannot enter this protocol')
        cameras[candidate['name']] = dict(
            transform=rigid_matrix(np.asarray(candidate['rotation'])).tolist(),
            target_intrinsic=intrinsic_matrix(np.asarray(candidate['intrinsic'])).tolist())
    return dict(schema='ms2_intrinsic_confirmation_plan_v1', sequence=base['sequence'],
                frame_count=base['frame_count'], frame_ids=quarter_frames(base['frame_ids']),
                previous_frames=base['frame_ids']+interval_midpoints(base['frame_ids']),
                selection='integer midpoint between each original left endpoint and its midpoint',
                split='new frames from the same official training sequence, not another scene/rig',
                prior_plan_sha256=BASE_SHA256, candidate_report_sha256=CANDIDATE_SHA256,
                candidates=cameras, thermal_window_dn=[3308., 4974.],
                wrong_pair_cyclic_offset=7, source_stereo='unchanged native RGB StereoSGBM',
                matcher_input_max_side=640, refit_allowed=False, frame_exclusion_allowed=False,
                fixed_denominator='author-calibration-supported points; lost support is failure',
                score_thresholds_target_native_px=[1., 3., 5., 10.],
                thresholds_are_qualification_gate=False, registration_qualified=False,
                generator_training_approved=False, antiuav_status_changed=False)


def checked_plan(path):
    if file_sha256(BASE_PATH) != BASE_SHA256 or file_sha256(CANDIDATE_PATH) != CANDIDATE_SHA256:
        raise ValueError('frozen input or candidate changed')
    expected = make_plan(json.loads(BASE_PATH.read_text()), json.loads(CANDIDATE_PATH.read_text()))
    if json.loads(path.read_text()) != expected:
        raise ValueError('plan differs from frozen full protocol')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'extract'))
    parser.add_argument('--plan', type=Path, default=PLAN_PATH)
    parser.add_argument('--output-root', type=Path,
                        default=Path('experiments/ms2_intrinsic_confirmation_sync_01'))
    parser.add_argument('--out', type=Path,
                        default=Path('experiments/ms2_intrinsic_confirmation_sync_01_report.json'))
    args = parser.parse_args()
    if file_sha256(BASE_PATH) != BASE_SHA256 or file_sha256(CANDIDATE_PATH) != CANDIDATE_SHA256:
        raise ValueError('frozen input or candidate changed')
    if args.action == 'plan':
        plan = make_plan(json.loads(BASE_PATH.read_text()), json.loads(CANDIDATE_PATH.read_text()))
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open('x') as handle:
            json.dump(plan, handle, indent=2, allow_nan=False)
        print(f'wrote {args.plan}; candidates frozen before image extraction')
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    plan = checked_plan(args.plan)
    if file_sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError('archive changed')
    initial_hashes = {str(p): file_sha256(p) for p in (
        args.plan, Path(__file__), Path('src/aero_ir/data/ms2_archive.py'),
        Path('scripts/prepare_ms2_confirmation.py'))}
    result = extract_screen(ARCHIVE, args.output_root, plan, 'sync_data', 24150818939)
    for path, digest in initial_hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'protocol or source changed during extraction: {path}')
    result.update(plan_sha256=initial_hashes[str(args.plan)], source_sha256=initial_hashes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out}; no qualification or generator approval')
    if not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
