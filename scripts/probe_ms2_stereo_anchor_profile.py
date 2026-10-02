"""Compare vertical score profiles at LiDAR, SGBM and RAFT horizontal anchors on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.raft_stereo import rectify_images
from aero_ir.registration.stereo_anchor_profile import (
    aggregate,
    known_shift_control,
    summarize,
    vertical_profile,
)
from aero_ir.registration.stereo_patch_profile import OFFSETS
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_raft_stereo import BASE, ROOT
from scripts.probe_ms2_stereo_depth import read

SOURCE = Path("experiments/ms2_stereo_patch_profile_01/report.json")
SOURCE_SHA = "f278c66d0d0b2c36635f95bfdd4be3c11d6bef673d44770bf0afce2dfd390fe0"
PATCH = Path("experiments/ms2_stereo_patch_evidence_01")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("frozen profile experiment changed")
    source = json.loads(SOURCE.read_text())
    hashes = source["input_and_source_sha256"].copy()
    for row in source["rows"]:
        path = SOURCE.parent / row["artifact"]
        if path.resolve().parent != SOURCE.parent.resolve():
            raise ValueError("unsafe source artifact")
        hashes[str(path)] = row["sha256"]
    hashes[str(SOURCE.parent / "preflight.json")] = source["preflight_sha256"]
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed source input: {path}")
    for path in (SOURCE, Path(__file__), Path("src/aero_ir/registration/stereo_anchor_profile.py")):
        hashes[str(path)] = file_sha256(path)
    baseline = json.loads(BASE.read_text())
    labels = ("sensor", "frame", "condition", "right_frame", "geometry_available")
    if [{k: r[k] for k in labels} for r in source["rows"]] != [
        {k: r[k] for k in labels} for r in baseline["rows"]
    ]:
        raise ValueError("case labels differ")
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = dict(
        schema="ms2_stereo_anchor_profile_plan_v1",
        input_and_source_sha256=hashes,
        anchors=["lidar", "sgbm", "raft"],
        vertical_offsets_px=OFFSETS.tolist(),
        patch_size=9,
        all_references_retained=True,
        all_offsets_and_anchors_common=True,
        synthetic_controls=dict(
            sample="64 evenly spaced original reference indices per case",
            disparity=6,
            vertical_shifts=[0, 1],
            texture="actual left image",
        ),
        model_inference=False,
        calibration_fit=False,
        registration_qualified=False,
        generator_training_approved=False,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(plan, indent=2, allow_nan=False))
    rows = []
    for i, (row, ref) in enumerate(zip(source["rows"], baseline["rows"], strict=True)):
        with np.load(PATCH / row["artifact"], allow_pickle=False) as f:
            patch = dict(f)
        with np.load(SOURCE.parent / row["artifact"], allow_pickle=False) as f:
            old = dict(f)
        with np.load(BASE.parent / ref["artifact"], allow_pickle=False) as f:
            original = dict(f)
        np.testing.assert_array_equal(patch["source_xy"], old["source_xy"])
        geometry = {
            k.removeprefix("geometry_"): v for k, v in original.items() if k.startswith("geometry_")
        }
        profiles = dict(lidar=old["ncc_profile_y"])
        controls, arrays = [], {}
        if geometry:
            images = []
            for side, frame in (("left", row["frame"]), ("right", row["right_frame"])):
                p = ROOT / row["sensor"] / f"img_{side}" / f"{frame}.png"
                if str(p) not in hashes or file_sha256(p) != hashes[str(p)]:
                    raise ValueError("unknown input image")
                raw = read(p)
                images.append(
                    cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
                    if row["sensor"] == "rgb"
                    else display_thermal(raw, (3308.0, 4974.0))
                )
            images, support = rectify_images(*images, geometry)
            for name in ("sgbm", "raft"):
                profiles[name] = vertical_profile(
                    *images, *support, patch["rectified_xy"], patch[f"disparity_{name}"]
                )
            indices = np.unique(np.linspace(0, len(patch["source_xy"]) - 1, 64, dtype=int))
            arrays["control_indices"] = indices
            for shift in (0, 1):
                scores, stats = known_shift_control(
                    images[0], support[0], patch["rectified_xy"][indices], shift
                )
                arrays[f"control_y{shift}"] = scores
                controls.append(stats)
                if not stats["ok"]:
                    raise ValueError(f"known-shift control failed: {row['artifact']}, {shift}")
        else:
            for name in ("sgbm", "raft"):
                profiles[name] = np.full_like(profiles["lidar"], np.nan)
        for name in profiles:
            np.testing.assert_allclose(
                profiles[name][:, len(OFFSETS) // 2],
                patch[f"ncc_{name}"],
                atol=1e-12,
                rtol=0,
                equal_nan=True,
            )
        path = args.out_dir / row["artifact"]
        np.savez_compressed(
            path,
            **{f"profile_{k}": v for k, v in profiles.items()},
            **arrays,
            source_xy=patch["source_xy"],
            left_std_dn=patch["left_std_dn"],
        )
        rows.append(
            {k: row[k] for k in labels}
            | dict(
                artifact=path.name,
                sha256=file_sha256(path),
                controls=controls,
                strata=summarize(profiles, patch["left_std_dn"]),
            )
        )
        print(f"{i + 1}/{len(source['rows'])} {path.stem}", flush=True)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during analysis: {p}")
    result = dict(
        schema="ms2_stereo_anchor_profile_v1",
        rows=rows,
        aggregate=aggregate(rows),
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
        note="Image evidence conditional on fixed horizontal anchors. Real-texture "
        "known shifts validate calculations, not cross-sensor physical accuracy. "
        "No camera correction, dense GT or qualified export is produced.",
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}; not registration qualification")


if __name__ == "__main__":
    main()
