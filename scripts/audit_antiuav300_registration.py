#!/usr/bin/env python3
"""Fit and audit a train-only Anti-UAV300 RGB-to-IR target-box transform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.antiuav_registration import build_registration_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/antiuav300_registration_audit.json"),
    )
    args = parser.parse_args()

    print("Fitting on Anti-UAV300/train; validating without refitting...", flush=True)
    report = build_registration_audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
