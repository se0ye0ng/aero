"""Assemble the RFS vector and its scalar aggregate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aero_ir.rfs.distances import DISTANCES, reference_floor
from aero_ir.rfs.stats import COMPONENTS, image_set_statistics


@dataclass
class RFSReport:
    """Per-component normalised distances plus the weighted scalar.

    ``per_component`` values are in units of the real-set sampling floor, so 1.0 means
    "as far from real as real is from itself". ``scalar`` is the weighted mean.
    """

    per_component: dict[str, float] = field(default_factory=dict)
    floors: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    scalar: float = float("nan")
    distance: str = "wasserstein"

    def to_dict(self) -> dict:
        return {
            "distance": self.distance,
            "scalar": self.scalar,
            "per_component": self.per_component,
            "floors": self.floors,
            "weights": self.weights,
            "component_names": COMPONENTS,
        }

    def worst(self, n: int = 3) -> list[tuple[str, float]]:
        """The components the generator failed hardest - the actionable part of the report."""
        items = [(k, v) for k, v in self.per_component.items() if np.isfinite(v)]
        return sorted(items, key=lambda kv: -kv[1])[:n]


def compute_rfs(real_set, gen_set, cfg) -> RFSReport:
    """Compare a generated set against a real set.

    Args:
        real_set / gen_set: ``(images, boxes_per_image)`` pairs.
        cfg: RFS config (see ``configs/rfs/default.yaml``).
    """
    dist_name = getattr(cfg, "distance", "wasserstein")
    distance = DISTANCES[dist_name]

    real_stats = image_set_statistics(real_set[0], real_set[1], cfg)
    gen_stats = image_set_statistics(gen_set[0], gen_set[1], cfg)

    per_component: dict[str, float] = {}
    floors: dict[str, float] = {}
    for key, real_vals in real_stats.items():
        floor = reference_floor(real_vals, distance)
        raw = distance(gen_stats.get(key, np.asarray([])), real_vals)
        floors[key] = floor
        per_component[key] = (
            float(raw / floor) if floor and np.isfinite(floor) and floor > 0 else float("nan")
        )

    weight_cfg = getattr(cfg, "weights", "uniform")
    if weight_cfg == "uniform":
        weights = {k: 1.0 / max(len(per_component), 1) for k in per_component}
    else:
        total = sum(weight_cfg.values()) or 1.0
        weights = {k: v / total for k, v in weight_cfg.items()}

    finite = [(weights.get(k, 0.0), v) for k, v in per_component.items() if np.isfinite(v)]
    scalar = float(sum(w * v for w, v in finite) / max(sum(w for w, _ in finite), 1e-12))

    return RFSReport(
        per_component=per_component,
        floors=floors,
        weights=weights,
        scalar=scalar,
        distance=dist_name,
    )
