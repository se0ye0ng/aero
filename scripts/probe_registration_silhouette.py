"""Fail-closed diagnostic of explicitly reviewed paired masks and saved maps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aero_ir.registration.silhouette import silhouette_scores
from aero_ir.utils.manifest import file_sha256


def artifact(record, root):
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise ValueError("artifact requires path and sha256")
    path = (root / record["path"]).resolve()
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def validate_manifest(manifest, root):
    def named(value):
        return isinstance(value, str) and bool(value.strip())

    if manifest.get("schema_version") != 1 or manifest.get("split") != "train":
        raise ValueError("requires schema_version=1 and split=train")
    if manifest.get("validation_or_test_access") != "none":
        raise ValueError("requires validation_or_test_access=none")
    review = manifest.get("review", {})
    if (
        review.get("approved") is not True
        or review.get("same_physical_outline") is not True
        or not named(review.get("reviewer_id"))
        or not named(review.get("reviewed_at"))
    ):
        raise ValueError(
            "human approval, reviewer, date, and same-physical-outline review required"
        )
    if not named(manifest.get("pair_id")):
        raise ValueError("explicit pair_id required")
    if (
        manifest.get("map_convention")
        != "target_to_source_normalized_pixel_centers_align_corners_false"
    ):
        raise ValueError("unsupported or missing coordinate convention")
    if manifest.get("candidate_generation_uses_masks") is not False:
        raise ValueError("candidates must be frozen independently of reviewed masks")
    checkpoint = artifact(manifest.get("checkpoint"), root)
    map_path = artifact(manifest.get("maps"), root)
    arrays = {}
    for side in ("source", "target"):
        record = manifest.get(side, {})
        if not named(record.get("frame_id")):
            raise ValueError(f"explicit {side} frame_id required")
        if record.get("image_grid") != "native":
            raise ValueError(
                f"{side} requires explicit native-image-grid masks; no implicit resizing"
            )
        image = artifact(record.get("image"), root)
        mask_path = artifact(record.get("mask"), root)
        arrays[side] = np.load(mask_path, allow_pickle=False)
        with Image.open(image) as frame:
            if arrays[side].shape != (frame.height, frame.width):
                raise ValueError(f"{side} reviewed mask does not match scoring-image grid")
    maps = np.load(map_path, allow_pickle=False)
    specs = manifest.get("candidate_specs")
    baseline = manifest.get("baseline_index")
    if (
        not isinstance(specs, list)
        or len(specs) != len(maps)
        or not isinstance(baseline, int)
        or isinstance(baseline, bool)
        or not 0 <= baseline < len(maps)
    ):
        raise ValueError("candidate specs and valid unchanged-map baseline_index required")
    if specs[baseline].get("is_unchanged") is not True:
        raise ValueError("baseline candidate must be explicitly marked unchanged")
    return arrays, maps, checkpoint


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    arrays, maps, checkpoint = validate_manifest(manifest, manifest_path.parent)
    result = silhouette_scores(arrays["source"], arrays["target"], maps)
    baseline = manifest["baseline_index"]
    # Never select from a changing set of usable candidates.
    complete = all(r["eligible"] for r in result["candidates"])
    selections = {}
    for metric in ("dice_loss", "symmetric_boundary_chamfer_px"):
        selected = None
        if complete:
            values = [r[metric] for r in result["candidates"]]
            if max(values) - min(values) > 1e-8:
                selected = (
                    baseline if values[baseline] <= min(values) + 1e-8 else int(np.argmin(values))
                )
        selections[metric] = selected
    root = Path(__file__).resolve().parents[1]
    return {
        "experiment": "reviewed_silhouette_candidate_diagnostic_v1",
        "manifest_sha256": file_sha256(manifest_path),
        "manifest": manifest,
        "checkpoint_sha256": file_sha256(checkpoint),
        "fit_split": "train",
        "validation_or_test_access": "none",
        "result": result,
        "selection": selections,
        "qualification": "hold_not_independent_pixel_correspondence",
        "sources": {
            p: file_sha256(root / p)
            for p in (
                "src/aero_ir/registration/silhouette.py",
                "scripts/probe_registration_silhouette.py",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior report")
    torch.set_num_threads(1)
    report = run(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
