"""Does the fidelity metric predict the downstream effect?

This is the evaluation that decides whether RFS is worth anything. Across every run in E3 and
E4, regress ``dAP`` on each candidate fidelity measure and compare predictive power.

H2 requires that the RFS scalar and vector beat FID and LPIPS. If they do not, that is the
result, and it is still worth reporting: it would establish that no cheap fidelity proxy
predicts training utility for infrared data.
"""

from __future__ import annotations

CANDIDATES = ["fid", "lpips", "ssim", "rfs_scalar", "rfs_vector"]


def evaluate(runs_table, target: str = "delta_ap") -> dict:
    """Return Spearman correlation and cross-validated R^2 per candidate.

    Args:
        runs_table: one row per run, with fidelity measures and the measured effect size.
    """
    raise NotImplementedError
