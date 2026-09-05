#!/usr/bin/env python3
"""Freeze the audited FLIR train/validation inputs into a replayable manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.flir import build_detector_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("experiments/flir_data_audit.json"),
    )
    parser.add_argument(
        "--representation",
        choices=("analytics16", "display8"),
        default="analytics16",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/flir_trainval_manifest.json"),
    )
    args = parser.parse_args()

    audit_report = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = build_detector_manifest(
        args.root,
        representation=args.representation,
        audit_report=audit_report,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "manifest_sha256": manifest["manifest_sha256"],
        "out": str(args.out),
        "representation": manifest["representation"],
        "splits": {split: value["counts"] for split, value in manifest["splits"].items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
