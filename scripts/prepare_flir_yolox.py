#!/usr/bin/env python3
"""Prepare and CPU-check the manifest-locked FLIR view consumed by YOLOX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.detect.flir_yolox import (
    build_flir_yolox_dataset,
    export_coco_subset,
    export_yolox_coco,
)
from aero_ir.detect.yolox_adapter import yolox_backend_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/flir_trainval_manifest.json"),
    )
    parser.add_argument(
        "--preprocess",
        type=Path,
        default=Path("experiments/flir_preprocess.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/flir_yolox"),
    )
    parser.add_argument("--smoke-train-images", type=int, default=128)
    parser.add_argument("--smoke-val-images", type=int, default=64)
    parser.add_argument("--smoke-seed", type=int, default=0)
    args = parser.parse_args()

    backend = yolox_backend_status()
    if not backend["available"]:
        raise RuntimeError(str(backend["reason"]))
    annotations = export_yolox_coco(args.manifest, args.out)
    smoke_annotations = {
        "train": export_coco_subset(
            args.out / "annotations" / "train.json",
            args.out / "annotations" / "smoke_train.json",
            n_images=args.smoke_train_images,
            seed=args.smoke_seed,
        ),
        "val": export_coco_subset(
            args.out / "annotations" / "val.json",
            args.out / "annotations" / "smoke_val.json",
            n_images=args.smoke_val_images,
            seed=args.smoke_seed,
        ),
    }
    samples: dict[str, dict] = {}
    for split in ("train", "val"):
        dataset = build_flir_yolox_dataset(
            image_root=args.root,
            prepared_root=args.out,
            annotation_file=f"{split}.json",
            preprocess_path=args.preprocess,
        )
        image = dataset.load_image(0)
        samples[split] = {
            "dataset_images": len(dataset),
            "first_image_shape": list(image.shape),
            "first_image_dtype": str(image.dtype),
            "first_image_min": int(image.min()),
            "first_image_max": int(image.max()),
        }
    report = {
        "backend": backend,
        "annotations": annotations,
        "smoke_annotations": smoke_annotations,
        "samples": samples,
        "gpu_used": False,
        "gate": "pass",
    }
    report_path = args.out / "preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
