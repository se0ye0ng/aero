"""Confirm frozen effective-camera corrections on new quarter-interval frames, on CPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from aero_ir.data.ms2_calibration import read_calibration
from aero_ir.data.ms2_timestamps import parse_timestamps
from aero_ir.registration.calibrated import from_millimetres, project_pixels, via_common_camera
from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.registration.ego_motion import interpolate_pose, moving_rig_transform
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.stereo_depth_diagnostic import compute_stereo, sample_disparity
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_confirmation import aggregate, compare_values
from scripts.audit_ms2_screen import verified_files
from scripts.prepare_ms2_intrinsic_confirmation import (
    ARCHIVE_SHA256,
    BASE_PATH,
    CANDIDATE_PATH,
    PLAN_PATH,
    checked_plan,
)
from scripts.probe_external_registration_landmarks import resize_gray, to_native
from scripts.probe_ms2_confirmation import add_identity, read
from scripts.probe_ms2_ego_motion import read_rgb_poses
from scripts.probe_ms2_stereo_depth import summarize
from scripts.probe_registration_rgb_resolution import ignored_bytecode_paths


def score_cameras(source, target, depth, source_k, target_k, transform, cameras,
                  source_shape=(384, 1224), target_shape=(256, 640)):
    reference = project_pixels(source, depth, source_k, target_k, transform,
                               source_shape=source_shape, target_shape=target_shape)
    supported = reference.supported
    scores = {}
    for name, camera in cameras.items():
        projection = project_pixels(source, depth, source_k, np.asarray(camera['target_intrinsic']),
                                    np.asarray(camera['transform']) @ transform,
                                    source_shape=source_shape, target_shape=target_shape)
        errors = np.linalg.norm(target-projection.target_xy, axis=1)
        errors[~supported | ~projection.supported] = np.inf
        scores[name] = dict(all_matches=summarize(errors),
                            fixed_supported=summarize(errors[supported]))
    return scores, supported


def load_protocol():
    plan = checked_plan(PLAN_PATH)
    if len(plan['frame_ids']) != 15 or set(plan['frame_ids']) & set(plan['previous_frames']):
        raise ValueError('unexpected or overlapping confirmation panel')
    config_path = Path('configs/experiment/registration_ms2_image_cpu.yaml')
    config = yaml.safe_load(config_path.read_text())
    base = json.loads(BASE_PATH.read_text())
    source = json.loads(CANDIDATE_PATH.read_text())
    hashes = source['input_and_source_sha256'].copy()
    root = Path('experiments/ms2_intrinsic_confirmation_sync_01')
    extraction_path = Path('experiments/ms2_intrinsic_confirmation_sync_01_report.json')
    extraction = json.loads(extraction_path.read_text())
    if extraction['archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('wrong extraction archive')
    for path, digest in extraction['source_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'extraction protocol/source changed: {path}')
    files = verified_files(root, extraction_path, PLAN_PATH, 'sync_data')
    for path in (*files.values(), config_path, BASE_PATH, CANDIDATE_PATH, PLAN_PATH,
                 extraction_path, Path(__file__),
                 Path('scripts/prepare_ms2_intrinsic_confirmation.py'),
                 Path('scripts/probe_ms2_confirmation.py'),
                 Path('scripts/analyze_ms2_confirmation.py')):
        add_identity(hashes, path)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'frozen dependency changed: {path}')
    seq = plan['sequence']
    calib = read_calibration(files[f'sync_data/{seq}/calib.npy'].read_bytes())
    author = plan['candidates']['author_calibration']
    if (not np.array_equal(author['target_intrinsic'], calib['K_thrL'])
            or not np.array_equal(author['transform'], np.eye(4))):
        raise ValueError('author camera differs from native reference')
    fixed = via_common_camera(from_millimetres(calib['R_nir2thr'], calib['T_nir2thr']),
                              from_millimetres(calib['R_nir2rgb'], calib['T_nir2rgb']))
    fb = calib['K_rgbL'][0, 0] * -calib['T_rgbR'].reshape(3)[0] / 1000.
    if (fb <= 0 or not np.allclose(calib['K_rgbL'], calib['K_rgbR'], atol=1e-9, rtol=0)
            or not np.allclose(calib['R_rgbR'], np.eye(3), atol=1e-9, rtol=0)
            or np.max(np.abs(calib['T_rgbR'].reshape(3)[1:])) > 1e-9):
        raise ValueError('unsupported stereo geometry')
    times = {s: parse_timestamps(files[f'sync_data/{seq}/{s}/img_left_timestamp.txt'].read_bytes(),
                                 expected_count=base['frame_count']) for s in ('rgb', 'thr')}
    right_times = parse_timestamps(
        files[f'sync_data/{seq}/rgb/img_right_timestamp.txt'].read_bytes(),
        expected_count=base['frame_count'])
    poses = read_rgb_poses(Path(config['odom']), base)
    transforms, pose_info = {}, {}
    for frame in plan['frame_ids']:
        pose, info = interpolate_pose(poses, times['rgb'], int(times['thr'][int(frame)]),
                                      max_extrapolation_ns=0)
        transforms[frame] = moving_rig_transform(fixed, poses[int(frame)], pose)
        pose_info[frame] = dict(pose_info=info,
                                rgb_stereo_skew_ns=int(right_times[int(frame)]
                                                      - times['rgb'][int(frame)]))
    return plan, config, hashes, files, calib, fb, transforms, pose_info


def verify(report_path, protocol):
    plan, _, hashes, _, calib, fb, transforms, _ = protocol
    report = json.loads(report_path.read_text())
    if (report['schema'] != 'ms2_intrinsic_confirmation_v1' or report['calibration_refitted']
            or report['plan_sha256'] != file_sha256(PLAN_PATH)
            or report['input_and_source_sha256'] != hashes):
        raise ValueError('changed inference protocol or inputs')
    preflight = report_path.parent/'preflight.json'
    if file_sha256(preflight) != report['preflight_sha256']:
        raise ValueError('changed inference preflight')
    # This also rejects absent/duplicate cases and changed candidate denominators.
    aggregate(report['rows'], plan['frame_ids'], plan['candidates'])
    frames = {x['frame']: x for x in report['frames']}
    if len(frames) != len(report['frames']) or set(frames) != set(plan['frame_ids']):
        raise ValueError('incomplete frame geometry')
    def artifact(name, digest):
        path = report_path.parent/name
        if (path.parent != report_path.parent
                or path.resolve().parent != report_path.parent.resolve()
                or file_sha256(path) != digest):
            raise ValueError('unsafe or changed artifact')
        hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as arrays:
            return dict(arrays)
    disparities = {frame: artifact(item['stereo_file'], item['stereo_sha256'])
                   for frame, item in frames.items()}
    for row in report['rows']:
        frame = row['frame']
        wrong = plan['frame_ids'][(plan['frame_ids'].index(frame)+7) % 15]
        if row['target_frame'] != (frame if row['condition'] == 'paired' else wrong):
            raise ValueError('wrong image pairing')
        arrays = artifact(row['artifact'], row['sha256'])
        disparity, _ = sample_disparity(disparities[frame], arrays['source_xy'])
        depth = fb/disparity
        if not np.allclose(depth, arrays['stereo_depth_m'], atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError('depth does not reproduce from stored stereo')
        scores, supported = score_cameras(arrays['source_xy'], arrays['target_xy'], depth,
                                          calib['K_rgbL'], calib['K_thrL'], transforms[frame],
                                          plan['candidates'])
        if not np.array_equal(supported, arrays['author_supported']):
            raise ValueError('fixed support changed')
        compare_values(row['scores'], scores)
    return dict(schema='ms2_intrinsic_confirmation_verified_v1',
                aggregate=aggregate(report['rows'], plan['frame_ids'], plan['candidates']),
                report_sha256=file_sha256(report_path), verified_artifact_sha256=hashes,
                registration_qualified=False, generator_training_approved=False,
                note='saved-array verification, not model replay or physical pixel GT')


def run(output, protocol):
    plan, config, hashes, files, calib, fb, transforms, pose_info = protocol
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    sys.pycache_prefix = tempfile.mkdtemp(prefix='aero-ms2-intrinsic-bytecode-')
    sys.dont_write_bytecode = True
    vendor = Path(config['vendor']).resolve()
    revision = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    if revision != config['vendor_commit']:
        raise ValueError('vendor changed')
    ignored = ignored_bytecode_paths(subprocess.check_output(
        ['git', '-C', str(vendor), 'status', '--porcelain', '--untracked-files=all'], text=True))
    preflight = dict(plan_sha256=file_sha256(PLAN_PATH), input_and_source_sha256=hashes,
                     thermal_window_dn=plan['thermal_window_dn'],
                     opencv_version=cv2.__version__, torch_version=torch.__version__,
                     device='cpu', refit=False, qualification=False)
    (output/'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
    prepared, frames = {}, []
    seq = plan['sequence']
    for frame in plan['frame_ids']:
        left, right = [read(files[f'sync_data/{seq}/rgb/img_{side}/{frame}.png'])
                       for side in ('left', 'right')]
        thermal = read(files[f'sync_data/{seq}/thr/img_left/{frame}.png'])
        if (left.shape != (384, 1224, 3) or left.dtype != np.uint8
                or right.shape != left.shape or right.dtype != left.dtype
                or thermal.shape != (256, 640) or thermal.dtype != np.uint16):
            raise ValueError('unexpected native images')
        gray, right_gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (left, right)]
        stereo = compute_stereo(gray, right_gray)
        path = output/f'stereo_{frame}.npz'
        np.savez_compressed(path, **stereo)
        prepared[frame] = dict(rgb=resize_gray(gray, 640), stereo=stereo,
                               thr=resize_gray(display_thermal(thermal, plan['thermal_window_dn']),
                                               640))
        frames.append(dict(frame=frame, stereo_file=path.name, stereo_sha256=file_sha256(path),
                           stereo_supported_pixels=int(stereo['valid'].sum()),
                           native_rgb_pixels=gray.size,
                           thermal_window_clipped_fraction=float(np.mean(
                               (thermal < plan['thermal_window_dn'][0])
                               | (thermal > plan['thermal_window_dn'][1]))), **pose_info[frame]))
    rows = []
    with torch.inference_mode():
        for name, spec in config['models'].items():
            model = load_matcher('xoftr', Path(spec['path']), vendor, 'cpu')
            for index, frame in enumerate(plan['frame_ids']):
                source = prepared[frame]
                wrong = plan['frame_ids'][(index+7) % 15]
                for condition, target_frame in (('paired', frame), ('unrelated_thermal', wrong)):
                    target = prepared[target_frame]
                    a, b, confidence, flags = infer(model, 'xoftr', source['rgb'][0],
                                                    target['thr'][0], 'cpu')
                    a, b = to_native(a, source['rgb'][1]), to_native(b, target['thr'][1])
                    disparity, _ = sample_disparity(source['stereo'], a)
                    depth = fb/disparity
                    scores, supported = score_cameras(a, b, depth, calib['K_rgbL'], calib['K_thrL'],
                                                      transforms[frame], plan['candidates'])
                    path = output/f'{name}_{frame}_{condition}.npz'
                    np.savez_compressed(path, source_xy=a, target_xy=b, confidence=confidence,
                                        stereo_depth_m=depth, author_supported=supported)
                    rows.append(dict(model=name, frame=frame, condition=condition,
                                     target_frame=target_frame, flags=flags, scores=scores,
                                     artifact=path.name, sha256=file_sha256(path)))
                print(f'{name} {frame}: frozen cameras + wrong-pair control complete', flush=True)
            del model
    aggregate(rows, plan['frame_ids'], plan['candidates'])
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'input changed during inference: {path}')
    return dict(schema='ms2_intrinsic_confirmation_v1', rows=rows, frames=frames,
                plan_sha256=file_sha256(PLAN_PATH), input_and_source_sha256=hashes,
                preflight_sha256=file_sha256(output/'preflight.json'),
                vendor_bytecode_ignored=ignored, device='cpu', calibration_refitted=False,
                registration_qualified=False, generator_training_approved=False,
                antiuav_status_changed=False,
                note='new frame IDs, same training sequence; stereo estimates, not pixel GT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('infer', 'verify'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'verify' and args.report is None:
        parser.error('verify requires --report')
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    protocol = load_protocol()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.action == 'infer':
        result = run(args.out_dir, protocol)
    else:
        result = verify(args.report, protocol)
    with (args.out_dir/'report.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out_dir / "report.json"}; not registration qualification')


if __name__ == '__main__':
    main()
