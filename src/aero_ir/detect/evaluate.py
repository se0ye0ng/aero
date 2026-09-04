"""COCO metrics plus the effect size the hypotheses are actually about.

Reporting raw AP per arm is not enough: every hypothesis concerns ``dAP``, the change caused
by adding generated data at a matched budget. Reporting it directly, with a bootstrap
interval over seeds, keeps the analysis honest about run-to-run spread.
"""

from __future__ import annotations

import numpy as np


def coco_metrics(predictions, ground_truth) -> dict[str, float]:
    """mAP/mAR at 0.5 and 0.5:0.95, overall and per class.

    Deliberately the same metric set as the results being tested, so numbers are directly
    comparable rather than merely similar.
    """
    raise NotImplementedError


def delta_ap(ap_with_gen: np.ndarray, ap_real_only: np.ndarray, n_boot: int = 2000,
             seed: int = 0) -> dict[str, float]:
    """Effect size with a bootstrap confidence interval over seeds."""
    a = np.asarray(ap_with_gen, dtype=np.float64)
    b = np.asarray(ap_real_only, dtype=np.float64)
    point = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    boots = [
        float(rng.choice(a, a.size, replace=True).mean() - rng.choice(b, b.size, replace=True).mean())
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta_ap": point, "ci_low": float(lo), "ci_high": float(hi),
            "sign_consistent": bool(lo > 0 or hi < 0)}
