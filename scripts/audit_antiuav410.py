#!/usr/bin/env python3
"""Audit the local Anti-UAV410 external-evaluation release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.antiuav410 import audit_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/antiuav410_data_audit.json"),
    )
    parser.add_argument(
        "--skip-archive-crc",
        action="store_true",
        help="engineering-only shortcut; a formal manifest requires a CRC-checked audit",
    )
    args = parser.parse_args()

    print(
        "Auditing all extracted splits, then hashing and CRC-checking the source archive...",
        flush=True,
    )
    report = audit_release(
        args.root,
        archive=args.archive,
        verify_archive_crc=not args.skip_archive_crc,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    if report["gates"]["external_evaluation_eligible"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
