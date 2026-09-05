#!/usr/bin/env python3
"""Audit a local Anti-UAV300 release without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.antiuav import audit_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--skip-archive-crc", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("experiments/antiuav300_data_audit.json"))
    args = parser.parse_args()

    report = audit_release(
        args.root,
        archive=args.archive,
        verify_archive_crc=not args.skip_archive_crc,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
