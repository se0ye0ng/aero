"""Adapter for visible-to-infrared diffusion generation.

Wraps a published V2IR diffusion checkpoint behind :class:`aero_ir.generate.base.Generator`.
Weights are not vendored; ``configs/generator/diffv2ir.yaml`` points at a local checkpoint.
"""

from __future__ import annotations


class DiffV2IRGenerator:
    name = "diffv2ir"

    def __init__(self, checkpoint: str, guidance_scale: float = 7.5, steps: int = 50,
                 use_segmentation: bool = True, caption_source: str = "blip",
                 batch_size: int = 8) -> None:
        self.checkpoint = checkpoint
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.use_segmentation = use_segmentation
        self.caption_source = caption_source
        self.batch_size = batch_size

    def generate(self, sources, labels, **kwargs):
        """TODO: load the checkpoint and run conditioned sampling."""
        raise NotImplementedError
