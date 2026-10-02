"""Image-gradient corroboration of frozen cameras, without learned correspondences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from aero_ir.registration.calibrated import project_pixels
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.ngcc import scores
from aero_ir.registration.stereo_depth_diagnostic import compute_stereo, sample_disparity
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_confirmation import add_identity, read
from scripts.probe_ms2_intrinsic_confirmation import load_protocol


def eroded_common_support(masks):
    """Fixed intersection, eroded so Sobel never reads unsupported neighbors."""
    if not masks or any(m.shape != masks[0].shape or m.dtype != bool for m in masks):
        raise ValueError('same-shaped boolean masks required')
    shared = np.logical_and.reduce(masks)
    tensor = torch.from_numpy(shared.astype(np.float32))[None, None]
    return F.avg_pool2d(tensor, 3, stride=1, padding=1)[0, 0] >= 1.


def image_scores(reference, warped, support):
    """Return similarities with fixed support, or missing for insufficient texture."""
    if (reference.shape != warped.shape or reference.ndim != 4
            or reference.shape[:2] != (1, 1)):
        raise ValueError('matching 1,1,H,W grayscale tensors required')
    if support.shape != reference.shape[-2:] or support.dtype != torch.bool:
        raise ValueError('boolean spatial support required')
    if support.sum() < 32:
        return dict(signed_ngcc=None, absolute_ngcc=None, ngf_style_squared_cosine=None)
    losses, valid = scores(warped, reference, support)
    return {name: float((1-loss)[0]) if bool(valid[0]) and bool(torch.isfinite(loss[0])) else None
            for name, loss in losses.items()}


def sampling_grid(projection_xy, target_shape, output_shape):
    """Native pixel centres to align_corners=False coordinates; NaN samples stay outside."""
    h, w = target_shape
    points = np.asarray(projection_xy, dtype=np.float64)
    if points.shape != (int(np.prod(output_shape)), 2):
        raise ValueError('projection length differs from output grid')
    grid = (2*(points+.5)/[w, h]-1).reshape(*output_shape, 2)
    grid[~np.isfinite(grid).all(axis=-1)] = 2.
    return torch.from_numpy(grid.astype(np.float32))[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    plan, _, hashes, files, calib, fb, transforms, _ = load_protocol()
    for path in (Path(__file__), Path('src/aero_ir/registration/ngcc.py')):
        add_identity(hashes, path)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(input_and_source_sha256=hashes, candidates=plan['candidates'],
                     model_inference=False, camera_refit=False,
                     coordinate_system='warp thermal into native RGB, gradients AFTER warp',
                     support='RGB stereo four-neighbor support intersected with all camera '
                     'projections; erode one pixel for Sobel; no texture-based pixel exclusion',
                     conditions=['paired', 'unrelated_thermal_offset7'],
                     note='Shared subset diagnostic, not full-image coverage or qualification')
    (args.out_dir/'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
    seq = plan['sequence']
    thermal = {}
    for frame in plan['frame_ids']:
        raw = read(files[f'sync_data/{seq}/thr/img_left/{frame}.png'])
        if raw.dtype != np.uint16 or raw.shape != (256, 640):
            raise ValueError('invalid native thermal image')
        gray = display_thermal(raw, plan['thermal_window_dn'])
        thermal[frame] = torch.from_numpy(gray.copy())[None, None].float()/255.
    rows, supports = [], []
    with torch.inference_mode():
        for index, frame in enumerate(plan['frame_ids']):
            images = [read(files[f'sync_data/{seq}/rgb/img_{side}/{frame}.png'])
                      for side in ('left', 'right')]
            if any(im.shape != (384, 1224, 3) or im.dtype != np.uint8 for im in images):
                raise ValueError('invalid native RGB image')
            left, right = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images]
            stereo = compute_stereo(left, right)
            yy, xx = np.indices(left.shape)
            source = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
            disparity, stereo_valid = sample_disparity(stereo, source)
            depth = fb/disparity
            projections, masks = {}, {}
            for name, camera in plan['candidates'].items():
                projected = project_pixels(source, depth, calib['K_rgbL'],
                                           np.asarray(camera['target_intrinsic']),
                                           np.asarray(camera['transform']) @ transforms[frame],
                                           source_shape=left.shape, target_shape=(256, 640))
                projections[name] = sampling_grid(projected.target_xy, (256, 640), left.shape)
                masks[name] = projected.supported.reshape(left.shape)
            shared = eroded_common_support(list(masks.values()))
            identity_support = eroded_common_support([masks['author_calibration']])
            supports.append(dict(frame=frame, native_pixels=left.size,
                                 source_stereo_supported=int(stereo_valid.sum()),
                                 eroded_author_supported=int(identity_support.sum()),
                                 eroded_common_supported=int(shared.sum()),
                                 common_loss_from_author=int((identity_support & ~shared).sum()),
                                 candidate_supported={k: int(v.sum()) for k, v in masks.items()}))
            reference = torch.from_numpy(left.copy())[None, None].float()/255.
            wrong = plan['frame_ids'][(index+7) % 15]
            for condition, target_frame in (('paired', frame), ('unrelated_thermal', wrong)):
                for name, grid in projections.items():
                    warped = F.grid_sample(thermal[target_frame], grid,
                                           align_corners=False, mode='bilinear',
                                           padding_mode='zeros')
                    values = image_scores(reference, warped, shared)
                    rows.append(dict(frame=frame, target_frame=target_frame, condition=condition,
                                     candidate=name, supported_pixels=int(shared.sum()),
                                     scores=values))
            print(f'{frame}: fixed-camera image-structure controls complete', flush=True)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f'input changed during image-structure test: {path}')
    result = dict(schema='ms2_image_structure_corroboration_v1', rows=rows, supports=supports,
                  input_and_source_sha256=hashes,
                  preflight_sha256=file_sha256(args.out_dir/'preflight.json'),
                  registration_qualified=False, generator_training_approved=False,
                  note='No learned match coordinates enter scores; not independent physical GT. '
                  'Common geometry subset excludes missing depth and out-of-view regions; '
                  'all coverage loss remains reported. No qualification threshold is selected.')
    with (args.out_dir/'report.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out_dir / "report.json"}; image structure is corroboration only')


if __name__ == '__main__':
    main()
