"""Composable sensor chain."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn


class IRSensorPipeline(nn.Module):
    """Apply sensor stages in physical order.

    An empty stage list is the identity, which is the ``sensor=none`` arm of the protocol -
    the control that shows how much of any effect is attributable to the sensor model itself.
    """

    def __init__(self, stages: Iterable[nn.Module] | None = None) -> None:
        super().__init__()
        self.stages = nn.ModuleList(list(stages or []))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x

    def apply_numpy(self, images: Iterable[np.ndarray]) -> list[np.ndarray]:
        """Apply the chain to 2-D numpy images, preserving one output per input image."""
        try:
            reference = next(self.parameters())
        except StopIteration:
            reference = next(self.buffers(), torch.empty(0))
        device = reference.device
        outputs = []
        with torch.no_grad():
            for image in images:
                array = np.asarray(image)
                if array.ndim != 2:
                    raise ValueError(f"expected a 2-D image, got shape {array.shape}")
                tensor = torch.as_tensor(array, dtype=torch.float32, device=device)[None, None]
                outputs.append(self(tensor)[0, 0].detach().cpu().numpy())
        return outputs

    def __len__(self) -> int:
        return len(self.stages)
