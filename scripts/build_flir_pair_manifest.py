#!/usr/bin/env python3
"""Build the official FLIR video-pair manifest and registration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.flir_pairs import audit_video_pair_registration, build_video_pair_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--video-map", required=True, type=Path)
    parser.add_argument(
        "--release-audit",
        type=Path,
        default=Path("experiments/flir_data_audit.json"),
    )
    parser.add_argument(
        "--registration-out",
        type=Path,
        default=Path("experiments/flir_video_registration_audit.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/flir_video_pair_manifest.json"),
    )
    args = parser.parse_args()

    release_audit = json.loads(args.release_audit.read_text(encoding="utf-8"))
    registration = audit_video_pair_registration(args.root, args.video_map)
    manifest = build_video_pair_manifest(
        args.root,
        args.video_map,
        registration_audit=registration,
        release_audit=release_audit,
    )
    args.registration_out.parent.mkdir(parents=True, exist_ok=True)
    args.registration_out.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pair_manifest_sha256": manifest["pair_manifest_sha256"],
                "registration_audit_sha256": registration["registration_audit_sha256"],
                "counts": manifest["counts"],
                "registration_gates": registration["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.registration_out}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
