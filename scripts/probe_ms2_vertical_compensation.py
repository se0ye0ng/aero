"""Re-estimate SGBM with unchanged/positive/negative half-pixel right-image sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.stereo_vertical_compensation import estimate
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_cross_stereo import summarize_errors
from scripts.probe_ms2_lidar_stereo import aggregate
from scripts.probe_ms2_raft_stereo import BASE, ROOT, score
from scripts.probe_ms2_stereo_depth import read

SOURCE = Path("experiments/ms2_stereo_anchor_profile_01/report.json")
SOURCE_SHA = "a2c9232c965b82f333ea19795b4e8886d72de0208480a8c4738ab143b5f60fc3"
OFFSETS = (0.0, 0.5, -0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("frozen anchor evidence changed")
    source = json.loads(SOURCE.read_text())
    hashes = source["input_and_source_sha256"].copy()
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed source input: {path}")
    for p in (
        SOURCE,
        Path(__file__),
        Path("src/aero_ir/registration/stereo_vertical_compensation.py"),
    ):
        hashes[str(p)] = file_sha256(p)
    baseline = json.loads(BASE.read_text())
    focal = json.loads((BASE.parent / "preflight.json").read_text())[
        "equivalent_disparity_focal_px"
    ]
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(
        schema="ms2_vertical_compensation_plan_v1",
        input_and_source_sha256=hashes,
        right_sampling_offsets_px=OFFSETS,
        zero_must_reproduce_baseline=True,
        interpolation="OpenCV linear uint8; zero bypasses resampling",
        masks="both contributing source rows must be observed",
        unchanged_camera_and_Q=True,
        geometry_correction_installed=False,
        exploratory_same_observed_panel=True,
        all_reference_points_retained=True,
        registration_qualified=False,
        generator_training_approved=False,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, allow_nan=False))
    rows, common_rows = [], []
    for index, ref in enumerate(baseline["rows"]):
        with np.load(BASE.parent / ref["artifact"], allow_pickle=False) as f:
            original = dict(f)
        geometry = {
            k.removeprefix("geometry_"): v for k, v in original.items() if k.startswith("geometry_")
        }
        images = []
        if geometry:
            for side, frame in (("left", ref["frame"]), ("right", ref["right_frame"])):
                p = ROOT / ref["sensor"] / f"img_{side}" / f"{frame}.png"
                if str(p) not in hashes or file_sha256(p) != hashes[str(p)]:
                    raise ValueError("unknown or changed image")
                raw = read(p)
                images.append(
                    cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
                    if ref["sensor"] == "rgb"
                    else display_thermal(raw, (3308.0, 4974.0))
                )
        errors_by_offset = {}
        for offset in OFFSETS:
            stereo = estimate(*images, geometry, offset) if geometry else {}
            z, errors, scores = score(stereo, original, focal)
            if offset == 0:
                for key, value in stereo.items():
                    np.testing.assert_array_equal(value, original[key])
                np.testing.assert_allclose(
                    z, original["estimated_depth_m"], atol=1e-12, rtol=0, equal_nan=True
                )
                np.testing.assert_allclose(
                    errors,
                    original["equivalent_disparity_errors"],
                    atol=1e-12,
                    rtol=0,
                    equal_nan=True,
                )
            errors_by_offset[str(offset)] = errors
            path = (
                args.out_dir / f"{ref['sensor']}_{ref['frame']}_{ref['condition']}_dy{offset}.npz"
            )
            np.savez_compressed(
                path,
                **stereo,
                estimated_depth_m=z,
                equivalent_disparity_errors=errors,
                source_xy=original["source_xy"],
            )
            rows.append(
                {
                    k: ref[k]
                    for k in ("sensor", "frame", "condition", "right_frame", "geometry_available")
                }
                | dict(offset=offset, artifact=path.name, sha256=file_sha256(path), **scores)
            )
        common = np.logical_and.reduce([np.isfinite(v) for v in errors_by_offset.values()])
        common_rows.append(
            dict(
                sensor=ref["sensor"],
                frame=ref["frame"],
                condition=ref["condition"],
                all_reference_points=len(common),
                common_points=int(common.sum()),
                scores={k: summarize_errors(v, common) for k, v in errors_by_offset.items()},
            )
        )
        print(
            f"{index + 1}/{len(baseline['rows'])} {ref['sensor']} {ref['frame']} "
            f"{ref['condition']}: zero replay passed, +/-0.5 estimated",
            flush=True,
        )
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input changed during analysis: {path}")
    result = dict(
        schema="ms2_vertical_compensation_v1",
        rows=rows,
        common_support=common_rows,
        aggregate={str(o): aggregate([r for r in rows if r["offset"] == o]) for o in OFFSETS},
        zero_replayed_cases=len(baseline["rows"]),
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
        note="Exploratory image-space compensation on an observed training panel. "
        "Author geometry and Q are unchanged, not newly calibrated. "
        "Full reference counts and adverse pair controls retained. "
        "No dense GT or generator export approval.",
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}; not registration qualification")


if __name__ == "__main__":
    main()
