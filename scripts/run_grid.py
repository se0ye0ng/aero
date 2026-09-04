"""Expand an experiment's sweep block into concrete runs.

The grid lives in ``configs/experiment/*.yaml``, never in this script. This file only
enumerates it, so that what was run is always readable from configuration.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml


def expand(sweep: dict) -> list[dict]:
    keys = list(sweep)
    return [dict(zip(keys, values, strict=True))
            for values in itertools.product(*(sweep[k] for k in keys))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="name under configs/experiment, without .yaml")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.configs) / "experiment" / f"{args.experiment}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    runs = expand(cfg.get("sweep", {}))

    print(f"{args.experiment}: {len(runs)} runs")
    for i, run in enumerate(runs):
        overrides = " ".join(f"{k}={v}" for k, v in run.items())
        cmd = f"python -m aero_ir.cli run experiment={args.experiment} {overrides}"
        print(f"[{i:03d}] {cmd}")
        if not args.dry_run:
            pass  # TODO: dispatch via hydra multirun or a job launcher


if __name__ == "__main__":
    main()
