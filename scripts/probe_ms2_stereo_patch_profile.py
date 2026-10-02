"""CPU fixed-axis local score profiles at every original LiDAR reference point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.raft_stereo import rectify_images
from aero_ir.registration.stereo_patch_profile import OFFSETS, aggregate, axis_profiles, summarize
from aero_ir.utils.manifest import file_sha256
from scripts.probe_ms2_raft_stereo import BASE, ROOT
from scripts.probe_ms2_stereo_depth import read

SOURCE = Path("experiments/ms2_stereo_patch_evidence_01/report.json")
SOURCE_SHA = "74de1eaafdfcbf556b0538d131b381b8880808a742fb0e07f1e67f5f8227da56"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("frozen patch evidence changed")
    source = json.loads(SOURCE.read_text())
    hashes = source["input_and_source_sha256"].copy()
    for row in source["rows"]:
        path = SOURCE.parent / row["artifact"]
        if path.resolve().parent != SOURCE.parent.resolve():
            raise ValueError("unsafe source artifact")
        hashes[str(path)] = row["sha256"]
    pre = SOURCE.parent / "preflight.json"
    hashes[str(pre)] = source["preflight_sha256"]
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed input: {p}")
    for p in (SOURCE, Path(__file__), Path("src/aero_ir/registration/stereo_patch_profile.py")):
        hashes[str(p)] = file_sha256(p)
    baseline = json.loads(BASE.read_text())
    labels = ("sensor", "frame", "condition", "right_frame", "geometry_available")
    if [{k: r[k] for k in labels} for r in source["rows"]] != [
        {k: r[k] for k in labels} for r in baseline["rows"]
    ]:
        raise ValueError("case inventory differs")
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(
        schema="ms2_stereo_patch_profile_plan_v1",
        input_and_source_sha256=hashes,
        offsets_rectified_px=OFFSETS.tolist(),
        patch_size=9,
        anchor="unchanged LiDAR-predicted right image position",
        axes="right x plus offset or right y plus offset, independently",
        joint_two_dimensional_search=False,
        whole_grid_support_required=True,
        separated_competitor_distance_px=1.0,
        tie_tolerance_ncc=1e-9,
        frames_or_points_dropped_by_score=False,
        registration_qualified=False,
        generator_training_approved=False,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, allow_nan=False))
    rows = []
    for index, (row, ref) in enumerate(zip(source["rows"], baseline["rows"], strict=True)):
        with np.load(SOURCE.parent / row["artifact"], allow_pickle=False) as f:
            previous = dict(f)
        with np.load(BASE.parent / ref["artifact"], allow_pickle=False) as f:
            original = dict(f)
        np.testing.assert_array_equal(previous["source_xy"], original["source_xy"])
        geometry = {
            k.removeprefix("geometry_"): v for k, v in original.items() if k.startswith("geometry_")
        }
        if geometry:
            images = []
            for side, frame in (("left", row["frame"]), ("right", row["right_frame"])):
                p = ROOT / row["sensor"] / f"img_{side}" / f"{frame}.png"
                if str(p) not in hashes or file_sha256(p) != hashes[str(p)]:
                    raise ValueError("unknown image input")
                raw = read(p)
                images.append(
                    cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
                    if row["sensor"] == "rgb"
                    else display_thermal(raw, (3308.0, 4974.0))
                )
            images, supports = rectify_images(*images, geometry)
            profiles = axis_profiles(
                *images, *supports, previous["rectified_xy"], previous["disparity_lidar"]
            )
        else:
            profiles = {
                f"ncc_profile_{a}": np.full((len(previous["source_xy"]), len(OFFSETS)), np.nan)
                for a in ("x", "y")
            }
        for axis in ("x", "y"):
            np.testing.assert_allclose(
                profiles[f"ncc_profile_{axis}"][:, len(OFFSETS) // 2],
                previous["ncc_lidar"],
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )
        path = args.out_dir / row["artifact"]
        np.savez_compressed(
            path,
            **profiles,
            source_xy=previous["source_xy"],
            left_std_dn=previous["left_std_dn"],
            offsets=OFFSETS,
        )
        rows.append(
            {k: row[k] for k in labels}
            | dict(
                artifact=path.name,
                sha256=file_sha256(path),
                strata=summarize(profiles, previous["left_std_dn"]),
            )
        )
        print(f"{index + 1}/{len(source['rows'])} {path.stem}", flush=True)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during analysis: {p}")
    result = dict(
        schema="ms2_stereo_patch_profile_v1",
        rows=rows,
        aggregate=aggregate(rows),
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
        note="Separate bounded axis searches, not full 2D matching or globally unique GT. "
        "Positive x offset decreases disparity. Peaks at endpoints are censored. "
        "No correction applied or qualification rule changed.",
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}; not registration qualification")


if __name__ == "__main__":
    main()
