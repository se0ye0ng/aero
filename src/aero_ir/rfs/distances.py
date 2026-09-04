"""Distributional distances between a generated and a real statistic."""

from __future__ import annotations

import numpy as np
from scipy import stats


def wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance. Default: scale-aware and robust to tail shape."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.wasserstein_distance(a, b))


def ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.ks_2samp(a, b).statistic)


def mmd(a: np.ndarray, b: np.ndarray, bandwidth: float | None = None) -> float:
    """Maximum mean discrepancy with an RBF kernel (median heuristic bandwidth)."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    if bandwidth is None:
        pooled = np.concatenate([a, b]).ravel()
        med = np.median(np.abs(pooled - np.median(pooled)))
        bandwidth = float(med) if med > 0 else 1.0

    def k(x, y):
        d2 = (x - y.T) ** 2
        return np.exp(-d2 / (2.0 * bandwidth**2))

    return float(k(a, a).mean() + k(b, b).mean() - 2.0 * k(a, b).mean())


DISTANCES = {"wasserstein": wasserstein, "ks": ks, "mmd": mmd}


def reference_floor(real: np.ndarray, distance, n_splits: int = 8, seed: int = 0) -> float:
    """Distance between two disjoint halves of the real set.

    This is the irreducible sampling floor. Normalising by it makes RFS interpretable:
    a value of 1.0 means the generated set is as far from real as real is from itself.
    """
    if real.size < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_splits):
        idx = rng.permutation(real.size)
        half = real.size // 2
        vals.append(distance(real[idx[:half]], real[idx[half : 2 * half]]))
    return float(np.nanmean(vals))
