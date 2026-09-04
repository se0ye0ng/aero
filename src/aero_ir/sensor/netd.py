"""Temporal detector noise parameterised by NETD.

NETD (noise-equivalent temperature difference) is the scene temperature difference that
produces a signal equal to the noise. Converting it to image units requires the scene
response in DN per kelvin, which is why both are configured together: an NETD figure alone
says nothing about how visible the noise is in an 8-bit image.
"""

from __future__ import annotations

import torch
from torch import nn


class NETDNoise(nn.Module):
    """Additive, spatially white, temporally independent Gaussian noise.

    Args:
        netd_mK: noise-equivalent temperature difference in millikelvin.
        scene_response_dn_per_K: digital numbers per kelvin of scene temperature.
        learnable: expose NETD as a parameter for sensor fitting.
    """

    def __init__(
        self,
        netd_mK: float = 50.0,
        scene_response_dn_per_K: float = 12.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        sigma_dn = (netd_mK / 1000.0) * scene_response_dn_per_K
        t = torch.tensor(float(sigma_dn))
        self.sigma_dn = nn.Parameter(t, requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self.sigma_dn.item() == 0.0:
            return x
        noise = torch.randn_like(x) * self.sigma_dn
        return x + noise

    def extra_repr(self) -> str:
        return f"sigma_dn={float(self.sigma_dn):.3f}"
