"""Pinned RAFT-Stereo inference adapter; learned disparity is not physical GT."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from aero_ir.utils.manifest import file_sha256

VENDOR = Path('experiments/external/raft_stereo_6e93ed2')
COMMIT = '6e93ed2169bd858dbb43033988563f3b0bb49506'
WEIGHTS = Path('experiments/external/raftstereo-middlebury.pth')
WEIGHTS_SHA = 'd22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819'
ITERATIONS = 32


def identities():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(VENDOR), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != COMMIT or git('diff', '--name-only', 'HEAD', '--'):
        raise ValueError('RAFT-Stereo vendor differs from reviewed commit')
    tracked = git('ls-tree', '-r', '--name-only', 'HEAD').splitlines()
    expected_py = {str(Path(p)) for p in tracked if p.endswith('.py')}
    present_py = {str(p.relative_to(VENDOR)) for p in VENDOR.rglob('*.py')}
    if present_py != expected_py:
        raise ValueError('unexpected vendor Python files')
    if file_sha256(WEIGHTS) != WEIGHTS_SHA:
        raise ValueError('RAFT-Stereo checkpoint differs from pinned official download')
    return {**{str(VENDOR/p): file_sha256(VENDOR/p) for p in tracked},
            str(WEIGHTS): WEIGHTS_SHA}


def load_model(device):
    identities()
    if device not in ('cpu', 'cuda'):
        raise ValueError('explicit cpu or cuda device required')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a valid GPU allocation, no CPU fallback')
    root = VENDOR.resolve()
    for name, module in list(sys.modules.items()):
        if name == 'core' or name.startswith('core.'):
            filename = getattr(module, '__file__', None)
            locations = list(getattr(module, '__path__', []))
            paths = ([Path(filename).resolve()] if filename
                     else [Path(p).resolve() for p in locations])
            if not paths or any(not p.is_relative_to(root) for p in paths):
                raise RuntimeError('core namespace belongs to another vendor; use a fresh process')
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        cls = importlib.import_module('core.raft_stereo').RAFTStereo
        padder = importlib.import_module('core.utils.utils').InputPadder
    finally:
        sys.path.remove(str(root))
        sys.dont_write_bytecode = old_bytecode
    args = SimpleNamespace(hidden_dims=[128]*3, context_norm='batch', n_downsample=2,
                           n_gru_layers=3, shared_backbone=False, corr_levels=4, corr_radius=4,
                           slow_fast_gru=False, mixed_precision=False, corr_implementation='reg')
    model = cls(args)
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(k.startswith('module.') for k in state):
        raise ValueError('unexpected official checkpoint format')
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    model.eval().to(device)
    return model, padder


def infer_disparity(model, padder_cls, left, right):
    if (left.dtype != np.uint8 or right.dtype != np.uint8 or left.shape != right.shape
            or left.ndim not in (2, 3) or (left.ndim == 3 and left.shape[2] != 3)):
        raise ValueError('equal uint8 grayscale or RGB images required')
    if min(left.shape[:2]) < 64:
        raise ValueError('image dimensions must be at least 64 pixels')
    device = next(model.parameters()).device
    tensors = []
    for image in (left, right):
        rgb = np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image
        tensors.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
                       .unsqueeze(0).to(device=device, dtype=torch.float32))
    padder = padder_cls(tensors[0].shape, divis_by=32)
    with torch.inference_mode():
        _, flow = model(*padder.pad(*tensors), iters=ITERATIONS, test_mode=True)
        # Official forward flow is x_right-x_left. Positive stereo disparity is its NEGATIVE.
        disparity = -padder.unpad(flow)[0, 0].float().cpu().numpy().astype(np.float64)
    if disparity.shape != left.shape[:2] or not np.isfinite(disparity).all():
        raise ValueError('invalid RAFT-Stereo output shape or nonfinite predictions')
    return disparity


def rectify_images(left, right, geometry):
    shape = tuple(int(v) for v in geometry['shape'])
    if (left.dtype != np.uint8 or right.dtype != np.uint8 or left.shape != right.shape
            or left.shape[:2] != shape):
        raise ValueError('native image geometry mismatch')
    h, w = shape
    images, supports = [], []
    for image, k, r, p in ((left, geometry['left_k'], geometry['r1'], geometry['p1']),
                           (right, geometry['right_k'], geometry['r2'], geometry['p2'])):
        mx, my = cv2.initUndistortRectifyMap(k, np.zeros(5), r, p[:, :3], (w, h), cv2.CV_32FC1)
        support = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (my >= 0)
        support &= (mx < w-1) & (my < h-1)
        # Retain the classical baseline's 5x5 remapping footprint for comparison.
        supports.append(cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool))
        images.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return images, supports


def consistency(disparity, reverse, left_support, right_support):
    """Same positive range (<128 px), reverse <=1px and border rules as baseline."""
    arrays = (disparity, reverse, left_support, right_support)
    if disparity.ndim != 2 or any(a.shape != disparity.shape for a in arrays):
        raise ValueError('equal 2D disparity and support arrays required')
    h, w = disparity.shape
    yy, xx = np.indices((h, w))
    finite = np.isfinite(disparity)
    xr = xx-np.where(finite, disparity, 0.)
    x0 = np.floor(np.clip(xr, -1, w)).astype(np.int64)
    clipped = np.clip(x0, 0, w-2)
    r0, r1 = reverse[yy, clipped], reverse[yy, clipped+1]
    blend = xr-x0
    sampled = r0*(1-blend)+r1*blend
    lr_error = np.abs(disparity-sampled)
    valid = finite & (disparity > 0) & (disparity < 128) & (x0 >= 0) & (x0+1 < w)
    valid &= (np.isfinite(r0) & np.isfinite(r1) & (r0 > 0) & (r1 > 0)
              & (r0 < 128) & (r1 < 128) & (lr_error <= 1.))
    valid &= left_support & right_support[yy, clipped] & right_support[yy, clipped+1]
    return dict(disparity=disparity, reverse_disparity=reverse, valid=valid, lr_error=lr_error,
                rectified_left_support=left_support, rectified_right_support=right_support)


def estimate_pair(model, padder, left, right, geometry):
    images, supports = rectify_images(left, right, geometry)
    disparity = infer_disparity(model, padder, *images)
    reverse = infer_disparity(model, padder, images[1][:, ::-1], images[0][:, ::-1])[:, ::-1]
    return consistency(disparity, reverse, *supports)
