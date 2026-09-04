"""Regenerate every figure and table from the runs table.

Figures are derived, never hand-edited. The headline figure is dAP against generated ratio,
split by initialisation - the single plot that shows whether the effect changes sign.
"""

from __future__ import annotations

import argparse

FIGURES = [
    "fig1_dap_vs_ratio_by_pretraining",      # E2 - the sign-flip test
    "fig2_dap_vs_ratio_by_budget_mode",      # E3 - substitution vs addition
    "fig3_fidelity_vs_dap_scatter",          # E4 - H1 and H2, FID against RFS
    "fig4_stratified_ap_by_target_area",     # F5 - where the effect actually lands
    "fig5_latency_vs_map_int8",              # deployment trade-off
]

TABLES = [
    "tab1_run_grid_summary",
    "tab2_predictive_power",                 # Spearman and cross-validated R^2 per metric
    "tab3_label_audit_exclusion_rates",      # F4 - reported with every result
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/runs.parquet")
    ap.add_argument("--out", default="experiments/report")
    args = ap.parse_args()
    raise SystemExit(
        f"TODO: build {len(FIGURES)} figures and {len(TABLES)} tables "
        f"from {args.runs} into {args.out}"
    )


if __name__ == "__main__":
    main()
