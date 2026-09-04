"""Radiometric Fidelity Score.

A diagnostic vector over physically meaningful image statistics, compared between a
generated set and a real set by distributional distance. See ``docs/rfs_spec.md``.

RFS earns its place only if it predicts downstream utility better than perceptual metrics do.
That evaluation lives in :mod:`aero_ir.analysis.predictive_power` and is part of the protocol,
not an afterthought.
"""

from aero_ir.rfs.report import RFSReport, compute_rfs
from aero_ir.rfs.stats import COMPONENTS, image_set_statistics

__all__ = ["COMPONENTS", "RFSReport", "compute_rfs", "image_set_statistics"]
