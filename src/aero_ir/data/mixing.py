"""Mixing real and generated data under an explicit budget.

This module exists because of F3. A single "generated ratio" number is ambiguous: raising it
can mean *removing real data* or *adding data on top*. Those two readings support opposite
conclusions, and conflating them is how a study concludes that generated data is harmful when
what it measured was that having less real data is harmful.

Both readings are implemented, and the protocol reports them separately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixSpec:
    """Resolved counts for one training arm."""

    n_real: int
    n_gen: int
    budget_mode: str
    gen_ratio: float

    @property
    def n_total(self) -> int:
        return self.n_real + self.n_gen

    @property
    def realised_ratio(self) -> float:
        return self.n_gen / self.n_total if self.n_total else 0.0


def resolve_mix(
    n_real_available: int,
    n_gen_available: int,
    gen_ratio: float,
    budget_mode: str = "fixed_total",
    total_images: int | None = None,
) -> MixSpec:
    """Turn a ratio into concrete counts.

    ``fixed_total``: total size is held constant; generated images displace real ones.
        Answers "is it safe to substitute generated for real data?"
    ``additive``: real count is held constant; generated images are added.
        Answers "does adding generated data on top help?"

    Raises:
        ValueError: on an out-of-range ratio, an unknown mode, or a request that the
            available pools cannot satisfy - failing loudly rather than silently
            delivering a different ratio than the one the experiment declared.
    """
    if not 0.0 <= gen_ratio <= 1.0:
        raise ValueError(f"gen_ratio must be in [0, 1], got {gen_ratio}")

    if budget_mode == "fixed_total":
        total = total_images if total_images is not None else n_real_available
        n_gen = int(round(total * gen_ratio))
        n_real = total - n_gen
    elif budget_mode == "additive":
        n_real = n_real_available if total_images is None else min(total_images, n_real_available)
        # gen_ratio is the generated fraction of the resulting total
        n_gen = 0 if gen_ratio >= 1.0 else int(round(n_real * gen_ratio / (1.0 - gen_ratio)))
    else:
        raise ValueError(f"unknown budget_mode: {budget_mode}")

    if n_real > n_real_available:
        raise ValueError(f"need {n_real} real images, only {n_real_available} available")
    if n_gen > n_gen_available:
        raise ValueError(f"need {n_gen} generated images, only {n_gen_available} available")

    return MixSpec(n_real=n_real, n_gen=n_gen, budget_mode=budget_mode, gen_ratio=gen_ratio)
