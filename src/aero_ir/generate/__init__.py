"""Generator adapters.

The repository does not propose a generative architecture. Generators are swappable and
declared in ``configs/generator/``; the contribution is the criterion applied to what they
produce and the protocol used to measure it.
"""

from aero_ir.generate.base import Generator
from aero_ir.generate.null_generator import NullGenerator
from aero_ir.generate.synthetic_baseline import SyntheticBaselineGenerator

__all__ = ["Generator", "NullGenerator", "SyntheticBaselineGenerator"]
