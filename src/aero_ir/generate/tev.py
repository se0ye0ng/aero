"""Temperature / emissivity / reflected-radiance conditioned generation.

An infrared image is not a picture of an object; it is a picture of radiance, which
decomposes roughly into emitted radiation scaled by emissivity plus reflected environmental
radiation. Conditioning a generator on that decomposition is established work, and it
improves image quality metrics.

What is *not* established is whether it improves the data's usefulness for training a
detector, because published physics-informed IR generation is evaluated on SSIM, PSNR, LPIPS
and FID and stops there. This adapter exists so that question can be answered on the same
grid as everything else in this repository.
"""

from __future__ import annotations


class TeVConditionedGenerator:
    name = "pid_tev"

    def __init__(self, backbone: str, checkpoint: str, tev: dict, loss: dict) -> None:
        self.backbone = backbone
        self.checkpoint = checkpoint
        self.tev = tev
        self.loss = loss

    def generate(self, sources, labels, **kwargs):
        """TODO: estimate the decomposition, condition sampling, recombine."""
        raise NotImplementedError
