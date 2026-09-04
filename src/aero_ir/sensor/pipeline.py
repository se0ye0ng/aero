"""Composable sensor chain."""

from __future__ import annotations

from collections.abc import Iterable

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

    def __len__(self) -> int:
        return len(self.stages)
