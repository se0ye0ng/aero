#!/usr/bin/env python3
"""Verify a run manifest and optionally execute its recorded replay command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.utils.manifest import verify_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the exact command recorded in this trusted local manifest",
    )
    args = parser.parse_args()
    report = verify_run_manifest(args.run, tolerance=args.tolerance, execute=args.execute)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
