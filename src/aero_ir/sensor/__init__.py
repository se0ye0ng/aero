"""Differentiable infrared sensor model.

Every stage is a ``torch.nn.Module`` so that the whole chain stays differentiable. That is a
deliberate design choice, not incidental: the N2 track back-propagates a detection loss
*through* this chain into the generator, so that generation is optimised for what the
detector actually consumes rather than for perceptual realism.

Stage order follows the physical signal chain::

    scene radiance -> optics (MTF) -> detector noise (NETD) -> non-uniformity (FPN/NUC)
                   -> gain control and quantisation (AGC, 8-bit)
"""

from aero_ir.sensor.agc import AGCQuantise
from aero_ir.sensor.fpn import FixedPatternNoise
from aero_ir.sensor.mtf import MTFBlur
from aero_ir.sensor.netd import NETDNoise
from aero_ir.sensor.pipeline import IRSensorPipeline

__all__ = ["AGCQuantise", "FixedPatternNoise", "MTFBlur", "NETDNoise", "IRSensorPipeline"]
