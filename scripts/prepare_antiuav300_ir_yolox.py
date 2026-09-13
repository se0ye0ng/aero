#!/usr/bin/env python3
"""Prepare or verify the native-IR Anti-UAV300 YOLOX engineering subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.antiuav300_ir import (
    export_native_ir_coco,
    prepare_native_ir_smoke,
    verify_prepared_native_ir_smoke,
)
from aero_ir.detect.antiuav300_yolox import build_antiuav300_yolox_dataset
from aero_ir.detect.yolox_adapter import yolox_backend_status
from aero_ir.utils.manifest import canonical_hash


def _sample_dataset(prepared_root: Path, split: str) -> dict:
    dataset = build_antiuav300_yolox_dataset(
        prepared_root=prepared_root,
        annotation_file=f"{split}.json",
    )
    first = dataset.load_image(0)
    last = dataset.load_image(len(dataset) - 1)
    return {
        "dataset_images": len(dataset),
        "first_image_shape": list(first.shape),
        "first_image_dtype": str(first.dtype),
        "first_image_min": int(first.min()),
        "first_image_max": int(first.max()),
        "last_image_shape": list(last.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("experiments/antiuav300_data_audit.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/antiuav300_ir_yolox"),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    backend = yolox_backend_status()
    if not backend["available"]:
        raise RuntimeError(str(backend["reason"]))
    if args.verify_only:
        report = verify_prepared_native_ir_smoke(args.out)
        samples = {split: _sample_dataset(args.out, split) for split in ("train", "val")}
        print(json.dumps({"preflight": report, "samples": samples}, indent=2, sort_keys=True))
        return
    if args.root is None:
        parser.error("--root is required unless --verify-only is used")

    audit_report = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = prepare_native_ir_smoke(args.root, args.out, audit_report=audit_report)
    manifest_path = args.out / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    annotations = export_native_ir_coco(manifest_path, args.out)
    samples = {split: _sample_dataset(args.out, split) for split in ("train", "val")}
    report = {
        "schema_version": 1,
        "kind": "antiuav300_native_ir_yolox_cpu_preflight",
        "manifest_sha256": manifest["manifest_sha256"],
        "backend": backend,
        "annotations": annotations,
        "samples": samples,
        "gpu_used": False,
        "scientific_status": "engineering_only_not_reportable",
        "gate": "pass",
    }
    report["preflight_sha256"] = canonical_hash(report)
    report_path = args.out / "preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_prepared_native_ir_smoke(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
