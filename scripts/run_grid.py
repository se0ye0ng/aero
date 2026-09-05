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
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(sweep[k] for k in keys))
    ]


def deduplicate_zero_generated(runs: list[dict], collapse_keys: list[str]) -> list[dict]:
    """Keep one canonical zero-generated control per remaining experimental condition."""
    if not collapse_keys:
        return runs
    unknown = set(collapse_keys) - {key for run in runs for key in run}
    if unknown:
        raise ValueError(f"zero-generated collapse keys are not in the sweep: {sorted(unknown)}")

    output: list[dict] = []
    seen_controls: set[tuple] = set()
    for run in runs:
        if float(run.get("mixing.gen_ratio", -1.0)) != 0.0:
            output.append(run)
            continue
        identity = tuple((key, value) for key, value in run.items() if key not in collapse_keys)
        if identity not in seen_controls:
            seen_controls.add(identity)
            output.append(run)
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="name under configs/experiment, without .yaml")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.configs) / "experiment" / f"{args.experiment}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cartesian_runs = expand(cfg.get("sweep", {}))
    collapse_keys = cfg.get("sweep_control", {}).get("collapse_at_zero_generated", [])
    runs = deduplicate_zero_generated(cartesian_runs, collapse_keys)

    removed = len(cartesian_runs) - len(runs)
    suffix = f" ({removed} duplicate zero-generated controls removed)" if removed else ""
    print(f"{args.experiment}: {len(runs)} runs{suffix}")
    for i, run in enumerate(runs):
        overrides = " ".join(f"{k}={v}" for k, v in run.items())
        cmd = f"python -m aero_ir.cli run experiment={args.experiment} {overrides}"
        print(f"[{i:03d}] {cmd}")
        if not args.dry_run:
            pass  # TODO: dispatch via hydra multirun or a job launcher


if __name__ == "__main__":
    main()
