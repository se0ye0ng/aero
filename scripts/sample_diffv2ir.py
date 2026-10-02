"""Isolated DiffV2IR inference worker. Does not train or approve any experiment."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
from PIL import Image

from aero_ir.generate.diffv2ir_sampling import (
    OfficialDiffV2IRSampler,
    sample_seed,
    source_identity,
    taming_source_identity,
)
from aero_ir.utils.manifest import file_sha256


def checked_path(root, item):
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_sha256(path) != item["sha256"]:
        raise ValueError("inference asset path/hash mismatch")
    return path


def read_image(root, item):
    path = checked_path(root, item)
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("inference conditioning must be explicitly rendered RGB")
        value = np.asarray(image).copy()
    if file_sha256(path) != item["sha256"]:
        raise ValueError("conditioning changed while decoding")
    return value


def run(request_path, expected_hash):
    request_path = request_path.resolve()
    if file_sha256(request_path) != expected_hash:
        raise ValueError("inference request hash mismatch")
    request = json.loads(request_path.read_text())
    if (
        request.get("kind") != "aero_diffv2ir_inference_request"
        or request.get("schema_version") != 1
    ):
        raise ValueError("unsupported inference request")
    root = request_path.parent
    project = Path(__file__).resolve().parents[1]
    required_sources = {
        "scripts/sample_diffv2ir.py",
        "src/aero_ir/generate/diffv2ir_adapter.py",
        "src/aero_ir/generate/diffv2ir_sampling.py",
    }
    source_hashes = request.get("bridge_source_sha256", {})
    if set(source_hashes) != required_sources:
        raise ValueError("incomplete sampling bridge source identity")
    for name, digest in source_hashes.items():
        if file_sha256(project / name) != digest:
            raise ValueError("sampling bridge differs from request")
    if (root / "result.json").exists() or (root / "generated").exists():
        raise FileExistsError("refusing to overwrite inference products")
    records = request["records"]
    ids = [r["source_id"] for r in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicate inference identities")
    for row in records:
        for role in ("rgb", "segmentation"):
            checked_path(root, row[role])
    # The worker has its own process environment; never changes registration's.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    backend = request["backend"]
    sampler = OfficialDiffV2IRSampler(**backend)
    (root / "generated").mkdir()
    results = []
    for index, row in enumerate(records):
        rgb, seg = (read_image(root, row[role]) for role in ("rgb", "segmentation"))
        image, geometry = sampler.sample(
            rgb,
            seg,
            row["prompt"],
            seed=sample_seed(request["seed"], row["source_id"]),
            **request["sampling"],
        )
        destination = root / "generated" / f"{index:06d}.png"
        with destination.open("xb") as handle:
            Image.fromarray(image).save(handle, format="PNG")
        results.append(
            {
                "source_id": row["source_id"],
                "path": str(destination.relative_to(root)),
                "sha256": file_sha256(destination),
                "geometry": geometry,
                "sample_seed": sample_seed(request["seed"], row["source_id"]),
                "label_validity": "unverified_requires_label_audit",
            }
        )
        print(f"DiffV2IR sampled {index + 1}/{len(records)}", flush=True)
    if file_sha256(request_path) != expected_hash:
        raise ValueError("inference request changed during sampling")
    if source_identity(Path(backend["upstream_root"])) != sampler.source:
        raise ValueError("upstream source changed during sampling")
    if taming_source_identity() != sampler.taming_source:
        raise ValueError("taming-transformers source changed during sampling")
    if file_sha256(backend["checkpoint"]) != backend["checkpoint_sha256"]:
        raise ValueError("checkpoint changed during sampling")
    for name, digest in source_hashes.items():
        if file_sha256(project / name) != digest:
            raise ValueError("sampling bridge changed during inference")
    for row in records:
        for role in ("rgb", "segmentation"):
            checked_path(root, row[role])
    for name, digest in sampler.clip_files.items():
        if file_sha256(Path(backend["clip_root"]) / name) != digest:
            raise ValueError("CLIP assets changed during sampling")
    import torch

    result = {
        "kind": "aero_diffv2ir_inference_outputs",
        "request_sha256": expected_hash,
        "upstream": sampler.source,
        "taming_upstream": sampler.taming_source,
        "clip_sha256": sampler.clip_files,
        "resolved_model_config": sampler.resolved_config,
        "outputs": results,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "installed_distributions": sorted(
                [distribution.metadata["Name"], distribution.version]
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            ),
        },
        "generator_training_eligible": "hold_not_qualified",
        "checkpoint_training_data_exposure": "not_established_by_inference",
        "output_radiometry": "display_rgb_not_calibrated_radiance",
    }
    with (root / "result.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    run(args.request, args.request_sha256)


if __name__ == "__main__":
    main()
