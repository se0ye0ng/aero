"""Fixed thermal sampling intervention at unchanged calibrated RGB-to-IR projections."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.calibrated import pixel_support
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.registration.stereo_depth_diagnostic import sample_disparity
from aero_ir.registration.stereo_vertical_compensation import estimate
from aero_ir.registration.temporal_stereo import native_to_rectified, sample_native_depth
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_cross_stereo import aggregate, expected_disparity, strata, summarize_errors
from scripts.probe_ms2_confirmation import add_identity
from scripts.probe_ms2_cross_stereo import depth_statistics
from scripts.probe_ms2_stereo_depth import read

BASE = Path("experiments/ms2_cross_stereo_02/report.json")
BASE_SHA = "c237aa8b490943a222276f52cb662d5c0ab646dfcecec34bafa8d4250625d794"
CANDIDATE = Path("experiments/ms2_vertical_confirmation_01/report.json")
CANDIDATE_SHA = "c8854f3945f77ad91be273af3fb1e88180af765c7dfddc6980d5eab24a4d89ef"
OFFSETS = (0.0, 0.5, -0.5)


def evaluate(stereo, geometry, reference, focal):
    xy, expected = reference["projected_xy"], reference["expected_z"]
    keep = reference["reference_supported"]
    eligible = keep & pixel_support(xy, (256, 640)) & np.isfinite(expected) & (expected > 0)
    estimated = np.full(len(keep), np.nan)
    estimated[eligible], _ = sample_native_depth(stereo, geometry, xy[eligible])
    depth_scores, _, supported = depth_statistics(expected, estimated, keep)
    observed, _ = sample_disparity(stereo, native_to_rectified(xy[eligible], geometry))
    predicted = expected_disparity(xy[eligible], expected[eligible], geometry)
    errors = np.full(len(keep), np.nan)
    errors[eligible] = (observed - predicted) * focal / geometry["p1"][0, 0]
    errors[~supported] = np.nan
    scores = {
        k: summarize_errors(errors, mask & keep)
        for k, mask in strata(reference["source_depth_m"]).items()
    }
    return estimated, errors, dict(depth_scores=depth_scores, strata=scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(BASE) != BASE_SHA or file_sha256(CANDIDATE) != CANDIDATE_SHA:
        raise ValueError("frozen evidence changed")
    baseline = json.loads(BASE.read_text())
    candidate = json.loads(CANDIDATE.read_text())
    hashes = baseline["input_and_source_sha256"].copy()
    for p, digest in candidate["input_and_source_sha256"].items():
        if p in hashes and hashes[p] != digest:
            raise ValueError(f"conflicting source identities: {p}")
        hashes[p] = digest
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed source: {p}")
    pre_path = BASE.parent / "preflight.json"
    if file_sha256(pre_path) != baseline["preflight_sha256"]:
        raise ValueError("baseline preflight changed")
    pre = json.loads(pre_path.read_text())
    focal = pre["cameras"]["author_calibration"]["target_intrinsic"][0][0]
    cases = [r for r in baseline["rows"] if r["camera"] == "author_calibration"]
    if len(cases) != 45 or len({(r["frame"], r["condition"]) for r in cases}) != 45:
        raise ValueError("incomplete author camera panel")
    for r in cases:
        p = BASE.parent / r["artifact"]
        if p.resolve().parent != BASE.parent.resolve() or file_sha256(p) != r["sha256"]:
            raise ValueError("changed or unsafe baseline artifact")
        add_identity(hashes, p)
    for p in (BASE, CANDIDATE, pre_path, Path(__file__)):
        add_identity(hashes, p)
    root = Path("experiments/ms2_intrinsic_confirmation_sync_01/sync_data/_2021-08-06-10-59-33")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = dict(
        input_and_source_sha256=hashes,
        offsets=list(OFFSETS),
        author_camera_only=True,
        rgb_depth_and_projection_unchanged=True,
        image_intervention="right thermal sampling, not camera recalibration",
        all_author_reference_points=True,
        equivalent_focal_px=focal,
        registration_qualified=False,
        generator_training_approved=False,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(plan, indent=2, allow_nan=False))
    rows, common = [], []
    for r in cases:
        with np.load(BASE.parent / r["artifact"], allow_pickle=False) as f:
            original = dict(f)
        geometry = {
            k.removeprefix("geometry_"): v for k, v in original.items() if k.startswith("geometry_")
        }
        images = []
        for side, frame in (("left", r["frame"]), ("right", r["right_frame"])):
            p = root / "thr" / f"img_{side}" / f"{frame}.png"
            if str(p) not in hashes or file_sha256(p) != hashes[str(p)]:
                raise ValueError("unrecorded image")
            images.append(display_thermal(read(p), (3308.0, 4974.0)))
        errors = {}
        for offset in OFFSETS:
            stereo = estimate(*images, geometry, offset)
            if offset == 0:
                for k, v in stereo.items():
                    np.testing.assert_array_equal(v, original[k])
            z, e, scores = evaluate(stereo, geometry, original, focal)
            if offset == 0:
                np.testing.assert_allclose(z, original["estimated_z"], rtol=0, atol=1e-12)
                if scores["depth_scores"] != r["scores"]:
                    raise ValueError("zero-offset baseline score changed")
            path = args.out_dir / f"{r['frame']}_{r['condition']}_dy{offset}.npz"
            np.savez_compressed(
                path,
                **stereo,
                estimated_z=z,
                disparity_errors=e,
                **{
                    k: original[k]
                    for k in (
                        "source_xy",
                        "source_depth_m",
                        "projected_xy",
                        "expected_z",
                        "reference_supported",
                    )
                },
                **{f"geometry_{k}": v for k, v in geometry.items()},
            )
            rows.append(
                dict(
                    frame=r["frame"],
                    condition=r["condition"],
                    right_frame=r["right_frame"],
                    camera=r["camera"],
                    offset=offset,
                    artifact=path.name,
                    sha256=file_sha256(path),
                    **scores,
                )
            )
            errors[str(offset)] = e
        mask = original["reference_supported"] & np.logical_and.reduce(
            [np.isfinite(e) for e in errors.values()]
        )
        common.append(
            dict(
                frame=r["frame"],
                condition=r["condition"],
                reference_points=int(original["reference_supported"].sum()),
                common_points=int(mask.sum()),
                scores={k: summarize_errors(e, mask) for k, e in errors.items()},
            )
        )
        print(
            f"{r['frame']} {r['condition']}: baseline replay and both offsets complete", flush=True
        )
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during run: {p}")
    result = dict(
        rows=rows,
        common_support=common,
        aggregate={str(o): aggregate([r for r in rows if r["offset"] == o]) for o in OFFSETS},
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
        limitation="Same-scene diagnostic. Equal-depth surfaces can agree at wrong "
        "correspondences. Stereo disparity discrepancy is NOT RGB-IR pixel error.",
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}; not qualification")


if __name__ == "__main__":
    main()
