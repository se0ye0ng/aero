"""Evaluate fixed LiDAR/SGBM/RAFT disparities against same-camera image patches on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.raft_stereo import rectify_images
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.registration.stereo_patch_evidence import aggregate, patch_evidence, summarize
from aero_ir.registration.temporal_stereo import native_to_rectified
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_cross_stereo import expected_disparity
from scripts.probe_ms2_raft_stereo import BASE, ROOT
from scripts.probe_ms2_stereo_depth import read

COMPARISON = Path("experiments/ms2_raft_stereo_comparison_01/report.json")
COMPARISON_SHA = "6634543db472ab2574bfcda03f10688c3858ddcf5442873d9a954c88fca3266a"
RAFT = Path("experiments/ms2_raft_stereo_01/report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(COMPARISON) != COMPARISON_SHA:
        raise ValueError("frozen verified estimator comparison changed")
    comparison = json.loads(COMPARISON.read_text())
    hashes = comparison["verified_artifact_sha256"].copy()
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed comparison dependency: {p}")
    for p in (
        COMPARISON,
        Path(__file__),
        Path("src/aero_ir/registration/stereo_patch_evidence.py"),
    ):
        hashes[str(p)] = file_sha256(p)
    baseline = json.loads(BASE.read_text())
    candidate = json.loads(RAFT.read_text())
    labels = ("sensor", "frame", "condition", "right_frame", "geometry_available")
    if [{k: r[k] for k in labels} for r in baseline["rows"]] != [
        {k: r[k] for k in labels} for r in candidate["rows"]
    ]:
        raise ValueError("case pairing differs from verified comparison")
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(
        schema="ms2_stereo_patch_plan_v1",
        input_and_source_sha256=hashes,
        cases=[{k: r[k] for k in labels} for r in baseline["rows"]],
        patch_size=9,
        interpolation="float64 bilinear, all four neighbours observed",
        objective="signed zero-mean normalized patch correlation",
        grayscale="RGB cv2.COLOR_RGB2GRAY; thermal 3308-4974 DN display window",
        texture_bins_left_std_dn=[2.0, 8.0],
        error_based_selection=False,
        constant_disparity_within_patch=True,
        patch_warp_optimized=False,
        registration_qualified=False,
        generator_training_approved=False,
        numpy_version=np.__version__,
        opencv_version=cv2.__version__,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, allow_nan=False))
    rows = []
    for index, (ref, new) in enumerate(zip(baseline["rows"], candidate["rows"], strict=True)):
        with np.load(BASE.parent / ref["artifact"], allow_pickle=False) as f:
            saved = dict(f)
        with np.load(RAFT.parent / new["artifact"], allow_pickle=False) as f:
            raft = dict(f)
        xy, depth = saved["source_xy"], saved["lidar_depth_m"]
        geometry = {
            k.removeprefix("geometry_"): v for k, v in saved.items() if k.startswith("geometry_")
        }
        rect_xy = np.full_like(xy, np.nan)
        ds = {k: np.full(len(xy), np.nan) for k in ("lidar", "sgbm", "raft")}
        if geometry:
            images = []
            for side, frame in (("left", ref["frame"]), ("right", ref["right_frame"])):
                path = ROOT / ref["sensor"] / f"img_{side}" / f"{frame}.png"
                if str(path) not in hashes or file_sha256(path) != hashes[str(path)]:
                    raise ValueError("unrecorded or changed image")
                raw = read(path)
                images.append(
                    cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
                    if ref["sensor"] == "rgb"
                    else display_thermal(raw, (3308.0, 4974.0))
                )
            rectified, support = rectify_images(*images, geometry)
            rect_xy = native_to_rectified(xy, geometry)
            ds["lidar"] = expected_disparity(xy, depth, geometry)
            ds["sgbm"], _ = sample_disparity(saved, rect_xy)
            ds["raft"], _ = sample_disparity(raft, rect_xy)
            evidence = patch_evidence(*rectified, *support, rect_xy, ds)
        else:
            evidence = dict(
                left_std_dn=np.full(len(xy), np.nan),
                **{f"ncc_{k}": np.full(len(xy), np.nan) for k in ds},
                **{f"patch_supported_{k}": np.zeros(len(xy), bool) for k in ds},
            )
        path = args.out_dir / f"{ref['sensor']}_{ref['frame']}_{ref['condition']}.npz"
        np.savez_compressed(
            path,
            source_xy=xy,
            lidar_depth_m=depth,
            rectified_xy=rect_xy,
            **{f"disparity_{k}": v for k, v in ds.items()},
            **evidence,
        )
        rows.append(
            {k: ref[k] for k in labels}
            | dict(artifact=path.name, sha256=file_sha256(path), strata=summarize(evidence))
        )
        print(f"{index + 1}/{len(baseline['rows'])} {path.stem}", flush=True)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during analysis: {p}")
    report = dict(
        schema="ms2_stereo_patch_evidence_v1",
        rows=rows,
        aggregate=aggregate(rows),
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
        note="Same-modality local photometric evidence, not independent physical GT. "
        "No subsetting by error, calibration fitting or model inference. "
        "LiDAR, stereo and patches share image geometry; NCC is not an independent "
        "metric of the photometric objective optimized by stereo estimators.",
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}; not registration qualification")


if __name__ == "__main__":
    main()
