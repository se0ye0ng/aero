#!/usr/bin/env python3
"""Freeze and export the Anti-UAV410 external test split for detection evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.antiuav410 import build_external_manifest, export_external_coco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("experiments/antiuav410_data_audit.json"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("experiments/antiuav410_test_manifest.json"),
    )
    parser.add_argument(
        "--coco-out",
        type=Path,
        default=Path("experiments/antiuav410_coco/test.json"),
    )
    args = parser.parse_args()

    print("Freezing the Anti-UAV410 external test manifest and COCO export...", flush=True)
    audit_report = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = build_external_manifest(args.root, audit_report=audit_report)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    coco = export_external_coco(args.manifest_out, args.coco_out)
    summary = {
        "manifest": str(args.manifest_out),
        "manifest_sha256": manifest["manifest_sha256"],
        "counts": manifest["splits"]["test"]["counts"],
        "exclusion_counts": manifest["splits"]["test"]["exclusion_counts"],
        "coco": coco,
        "gpu_used": False,
        "gate": "pass",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
