"""Selection policies, compared at an equal generated-sample budget.

The comparison that matters is ``PerceptualSelector`` against ``RFSSelector``: same budget,
same generator, same detector, differing only in what "good generated data" is taken to mean.
That single contrast tests H1 and H2 directly.
"""

from __future__ import annotations

import numpy as np


class NoSelection:
    """Use every generated sample. The control arm."""

    name = "none"

    def select(self, candidates, k=None, **kwargs):
        return list(range(len(candidates)))


class RandomSelector:
    """Uniform random selection. The baseline any policy must beat to be worth anything."""

    name = "random"

    def __init__(self, budget: float = 0.0, seed: int = 0) -> None:
        self.budget = budget
        self.rng = np.random.default_rng(seed)

    def select(self, candidates, k, **kwargs):
        n = len(candidates)
        k = min(int(k), n)
        return sorted(self.rng.choice(n, size=k, replace=False).tolist())


class PerceptualSelector:
    """Rank by a perceptual fidelity proxy (per-sample distance to the real feature centroid).

    This is the policy an FID-guided workflow implies. If it wins, H1 and H2 are refuted -
    which would itself be a clean, reportable result.
    """

    name = "perceptual"

    def __init__(self, metric: str = "fid", budget: float = 0.0) -> None:
        self.metric = metric
        self.budget = budget

    def select(self, candidates, k, **kwargs):
        raise NotImplementedError("TODO: per-sample perceptual proxy")


class RFSSelector:
    """Rank by radiometric consistency.

    ``mode="closest"`` keeps samples whose radiometric statistics sit closest to the real
    distribution. ``mode="coverage"`` instead maximises spread subject to a consistency
    floor, on the reasoning that generated data earns its place by covering scenarios the
    real set lacks - not by being maximally average.
    """

    name = "rfs"

    def __init__(self, budget: float = 0.0, components=None, mode: str = "closest") -> None:
        self.budget = budget
        self.components = components or ["R1", "R3", "R4", "R5", "R6"]
        self.mode = mode

    def select(self, candidates, k, real_reference=None, **kwargs):
        raise NotImplementedError("TODO: per-sample RFS then rank or greedy-cover")


class MarginalAPSelector:
    """Rank by estimated marginal contribution to validation AP.

    The strongest baseline and the most expensive. Included so the RFS result is reported
    against a utility-optimal reference rather than only against random selection.
    """

    name = "marginal_ap"

    def __init__(self, budget: float = 0.0, proxy: str = "gradient_alignment",
                 probe_steps: int = 200) -> None:
        self.budget = budget
        self.proxy = proxy
        self.probe_steps = probe_steps

    def select(self, candidates, k, **kwargs):
        raise NotImplementedError("TODO: gradient-alignment proxy or holdout probe")
