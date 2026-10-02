"""Sampling bridge to pinned official DiffV2IR code; optional, isolated runtime.

No upstream model implementation is vendored. The caller supplies local weights,
CLIP assets, frozen prompts and segmentation. This module does not train or qualify
registration, generate captions, or download dependencies/models automatically.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from aero_ir.utils.manifest import file_sha256

UPSTREAM_REVISION = "495947eaad0c2f3bb380ddf5f8820c2b1443e23f"
TAMING_REVISION = "3ba01b241669f5ade541ce990f7650a3b8f65318"


def source_identity(root: Path, *, expected_revision=UPSTREAM_REVISION, label="DiffV2IR") -> dict:
    root = root.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != expected_revision:
        raise ValueError(f"{label} checkout does not match the pinned upstream revision")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root).returncode:
        raise ValueError(f"{label} tracked source differs from the pinned revision")
    # Untracked Python files can shadow tracked modules even with a clean git diff.
    extras = (
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
        )
        .decode()
        .split("\0")
    )
    if any(p.endswith((".py", ".so", ".pyc")) and "__pycache__" not in p for p in extras):
        raise ValueError(f"untracked executable modules in the {label} checkout")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    return {"revision": revision, "tracked_sha256": {n: file_sha256(root / n) for n in names if n}}


def taming_source_identity():
    """The pinned upstream is a namespace package; an ordinary wheel is empty."""
    spec = importlib.util.find_spec("taming")
    paths = list(spec.submodule_search_locations or []) if spec else []
    if len(paths) != 1:
        raise ValueError("one pinned editable taming-transformers checkout is required")
    root = Path(paths[0]).resolve().parent
    return {"root": str(root), **source_identity(
        root, expected_revision=TAMING_REVISION, label="taming-transformers")}


def sampling_inputs(rgb, segmentation):
    """Pad only right/bottom to multiples of64; never resize/crop source objects."""
    for name, value in (("rgb", rgb), ("segmentation", segmentation)):
        if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
            raise ValueError(f"{name} must be an explicit uint8 image")
        if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) <= 0:
            raise ValueError(f"{name} must have nonempty H,W,3 shape")
    if rgb.shape != segmentation.shape:
        raise ValueError("RGB and segmentation must use the same observation grid")
    h, w = rgb.shape[:2]
    pad_h, pad_w = (-h) % 64, (-w) % 64
    tensors = []
    for image in (rgb, segmentation):
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        tensors.append(torch.from_numpy(padded.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1)
    return *tensors, {
        "input_size_hw": [h, w],
        "model_size_hw": [h + pad_h, w + pad_w],
        "padding_bottom_right": [pad_h, pad_w],
        "padding_mode": "edge",
        "output_geometry": "crop_padding_only; original_source_pixel_coordinates",
    }


def restore_prediction(prediction, geometry):
    h, w = geometry["input_size_hw"]
    mh, mw = geometry["model_size_hw"]
    if tuple(prediction.shape) != (1, 3, mh, mw) or not torch.isfinite(prediction).all():
        raise ValueError("invalid/nonfinite decoded diffusion output")
    value = ((prediction[0, :, :h, :w].float() + 1) * 127.5).clamp(0, 255).round()
    return value.permute(1, 2, 0).to(torch.uint8).cpu().numpy().copy()


def sample_seed(seed: int, source_id: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a nonempty string")
    digest = hashlib.sha256(f"{seed}\0{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**63


def official_modules(root: Path):
    """Import actual upstream modules in the isolated process, without its CLI."""
    root = root.resolve()
    taming_source_identity()
    sys.path[:0] = [str(root), str(root / "stable_diffusion")]
    try:
        import k_diffusion as K
        from omegaconf import OmegaConf

        spec = importlib.util.spec_from_file_location("aero_upstream_diffv2ir", root / "infer.py")
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        from ldm.util import instantiate_from_config
    except ImportError as error:
        raise RuntimeError(
            "Incomplete DiffV2IR runtime. Install/review dependencies in a separate "
            "environment; do not modify the active registration environment."
        ) from error
    return K, OmegaConf, upstream, instantiate_from_config


class OfficialDiffV2IRSampler:
    """Real model loader/sampler; run only in a dedicated dependency environment.

    CPU tests exercise the interface separately. Full upstream model loading and
    GPU sampling still require an independently verified checkpoint fixture.
    """

    def __init__(self, upstream_root, checkpoint, checkpoint_sha256, clip_root):
        root, checkpoint, clip_root = map(Path, (upstream_root, checkpoint, clip_root))
        self.source = source_identity(root)
        self.taming_source = taming_source_identity()
        if not checkpoint_sha256 or file_sha256(checkpoint) != checkpoint_sha256:
            raise ValueError("DiffV2IR checkpoint SHA256 mismatch or missing expected hash")
        if not clip_root.is_dir() or not (clip_root / "config.json").is_file():
            raise ValueError("local CLIP model/tokenizer directory is required")
        self.clip_files = {
            str(p.relative_to(clip_root)): file_sha256(p)
            for p in sorted(clip_root.rglob("*"))
            if p.is_file()
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no DiffV2IR model was loaded")
        K, OmegaConf, upstream, instantiate_from_config = official_modules(root)
        config = OmegaConf.load(root / "configs/generate.yaml")
        config.model.params.cond_stage_config.params = {
            "version": str(clip_root.resolve()),
            "device": "cuda",
        }
        if config.model.params.get("ckpt_path"):
            raise ValueError("unexpected implicit checkpoint path in inference config")
        self.resolved_config = OmegaConf.to_container(config, resolve=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["state_dict"]
        if not isinstance(state, dict) or not state:
            raise ValueError("expected nonempty DiffV2IR state_dict")
        if any(
            not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in state.values()
        ):
            raise ValueError("checkpoint has non-tensor or nonfinite model state")
        self.model = instantiate_from_config(config.model)
        # A stage1/base checkpoint is not silently upgraded to a stage2 sampler.
        self.model.load_state_dict(state, strict=True)
        self.model.eval().cuda()
        self.model.requires_grad_(False)
        self.denoiser = K.external.CompVisDenoiser(self.model)
        self.guided = upstream.CFGDenoiser(self.denoiser)
        self.euler = K.sampling.sample_euler_ancestral

    def sample(self, rgb, segmentation, prompt, *, seed, steps, text_scale, image_scale, seg_scale):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("a frozen nonempty prompt is required")
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be positive")
        if any(not math.isfinite(s) or s < 0 for s in (text_scale, image_scale, seg_scale)):
            raise ValueError("guidance scales must be finite and nonnegative")
        rgb_tensor, seg_tensor, geometry = sampling_inputs(rgb, segmentation)
        with torch.no_grad(), torch.autocast("cuda"), self.model.ema_scope():
            conditioned = {
                "c_crossattn": [self.model.get_learned_conditioning([prompt])],
                "c_concat1": [self.model.encode_first_stage(rgb_tensor.cuda()).mode()],
                "c_concat2": [self.model.encode_first_stage(seg_tensor.cuda()).mode()],
            }
            unconditioned = {
                "c_crossattn": [self.model.get_learned_conditioning([""])],
                "c_concat1": [torch.zeros_like(conditioned["c_concat1"][0])],
                "c_concat2": [torch.zeros_like(conditioned["c_concat2"][0])],
            }
            sigmas = self.denoiser.get_sigmas(steps)
            torch.manual_seed(seed)
            noise = torch.randn_like(conditioned["c_concat1"][0]) * sigmas[0]
            latent = self.euler(
                self.guided,
                noise,
                sigmas,
                extra_args={
                    "cond": conditioned,
                    "uncond": unconditioned,
                    "text_cfg_scale": text_scale,
                    "image_cfg_scale": image_scale,
                    "seg_cfg_scale": seg_scale,
                },
            )
            prediction = self.model.decode_first_stage(latent)
        return restore_prediction(prediction, geometry), geometry
