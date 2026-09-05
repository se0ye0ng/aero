#!/usr/bin/env python3
"""Audit a local Teledyne FLIR ADAS v2 release without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.flir import audit_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--video-map", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path("experiments/flir_data_audit.json"))
    args = parser.parse_args()

    report = audit_release(args.root, args.video_map, args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
