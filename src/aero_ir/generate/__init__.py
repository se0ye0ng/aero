"""Generator adapters.

The repository does not propose a generative architecture. Generators are swappable and
declared in ``configs/generator/``; the contribution is the criterion applied to what they
produce and the protocol used to measure it.
"""

from aero_ir.generate.base import Generator

__all__ = ["Generator"]
