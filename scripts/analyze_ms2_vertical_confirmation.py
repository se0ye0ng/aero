"""Verify the complete fixed-offset panel before summarizing same-point effects."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.calibrated import ms2_depth_metres
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_ms2_cross_stereo import summarize_errors
from scripts.probe_ms2_lidar_stereo import aggregate
from scripts.probe_ms2_raft_stereo import score
from scripts.probe_ms2_stereo_depth import read

OFFSETS = (0.0, 0.5, -0.5)
CONDITIONS = ("paired_static", "paired_timed", "unrelated_right_timed")


def require_complete_cases(rows, frames):
    expected = set(itertools.product(("rgb", "thr"), frames, CONDITIONS, OFFSETS))
    keys = [(r["sensor"], r["frame"], r["condition"], r["offset"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing, duplicate or unplanned case")


def common_effects(common):
    result = []
    for sensor, condition in itertools.product(("rgb", "thr"), CONDITIONS):
        rows = [r for r in common if r["sensor"] == sensor and r["condition"] == condition]
        available = [r for r in rows if r["common_points"]]
        baseline = np.array([r["scores"]["0.0"]["conditional_median_abs_px"] for r in available])
        for offset in OFFSETS:
            medians = np.array(
                [r["scores"][str(offset)]["conditional_median_abs_px"] for r in available]
            )
            result.append(
                dict(
                    sensor=sensor,
                    condition=condition,
                    offset=offset,
                    planned_frames=len(rows),
                    frames_with_common_points=len(available),
                    reference_points=sum(r["all_reference_points"] for r in rows),
                    common_points=sum(r["common_points"] for r in rows),
                    median_of_common_frame_medians_px=float(np.median(medians))
                    if len(medians)
                    else None,
                    improved_frame_count=int(np.count_nonzero(medians < baseline)),
                    worsened_frame_count=int(np.count_nonzero(medians > baseline)),
                    tied_frame_count=int(np.count_nonzero(medians == baseline)),
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    path = args.run / "report.json"
    report_sha = file_sha256(path)
    r = json.loads(path.read_text())
    pre_path = args.run / "preflight.json"
    if file_sha256(pre_path) != r["preflight_sha256"]:
        raise ValueError("preflight changed")
    pre = json.loads(pre_path.read_text())
    if pre["input_and_source_sha256"] != r["input_and_source_sha256"]:
        raise ValueError("source inventory changed")
    for p, digest in r["input_and_source_sha256"].items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed input: {p}")
    if r["registration_qualified"] or r["generator_training_approved"]:
        raise ValueError("unjustified approval")
    protocol = pre["protocol"]
    frames = protocol["frame_ids"]
    require_complete_cases(r["rows"], frames)
    references = {}
    depth_root = Path("experiments/ms2_vertical_confirmation_depth_01/proj_depth")
    for sensor, frame in itertools.product(("rgb", "thr"), frames):
        dp = depth_root / protocol["sequence"] / sensor / "depth" / f"{frame}.png"
        if str(dp) not in r["input_and_source_sha256"]:
            raise ValueError("unrecorded reference depth")
        depth = ms2_depth_metres(read(dp))
        yy, xx = np.nonzero(np.isfinite(depth))
        references[sensor, frame] = np.c_[xx, yy].astype(float), depth[yy, xx]
    errors, geometries, seen_artifacts = {}, {}, set()
    for row in r["rows"]:
        key = tuple(row[k] for k in ("sensor", "frame", "condition"))
        expected_right = frames[(frames.index(row["frame"]) + 7) % len(frames)]
        if row["right_frame"] != (
            expected_right if row["condition"] == "unrelated_right_timed" else row["frame"]
        ):
            raise ValueError("changed pairing")
        artifact = args.run / row["artifact"]
        if (
            artifact.resolve().parent != args.run.resolve()
            or artifact in seen_artifacts
            or file_sha256(artifact) != row["sha256"]
        ):
            raise ValueError("unsafe, repeated or changed artifact")
        seen_artifacts.add(artifact)
        with np.load(artifact, allow_pickle=False) as f:
            data = dict(f)
        xy, depth = references[row["sensor"], row["frame"]]
        np.testing.assert_array_equal(xy, data["source_xy"])
        np.testing.assert_array_equal(depth, data["lidar_depth_m"])
        geometry = {k: v for k, v in data.items() if k.startswith("geometry_")}
        if bool(geometry) != row["geometry_available"]:
            raise ValueError("geometry availability mismatch")
        if key in geometries:
            if geometry.keys() != geometries[key].keys():
                raise ValueError("geometry changed across offsets")
            for k, v in geometry.items():
                np.testing.assert_array_equal(v, geometries[key][k])
        else:
            geometries[key] = geometry
        z, e, scores = score(data, data, pre["equivalent_disparity_focal_px"])
        np.testing.assert_allclose(z, data["estimated_depth_m"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(e, data["equivalent_disparity_errors"], atol=1e-12, rtol=0)
        for k in ("strata", "depth_scores"):
            if scores[k] != row[k]:
                raise ValueError(f"score mismatch: {key} {k}")
        errors.setdefault(key, {})[str(row["offset"])] = e
    common = []
    for key, values in errors.items():
        mask = np.logical_and.reduce([np.isfinite(e) for e in values.values()])
        common.append(
            dict(
                sensor=key[0],
                frame=key[1],
                condition=key[2],
                all_reference_points=len(mask),
                common_points=int(mask.sum()),
                scores={o: summarize_errors(e, mask) for o, e in values.items()},
            )
        )
    if common != r["common_support"]:
        raise ValueError("common support summary mismatch")
    for offset in OFFSETS:
        if (
            aggregate([x for x in r["rows"] if x["offset"] == offset])
            != r["aggregate"][str(offset)]
        ):
            raise ValueError("aggregate mismatch")
    if file_sha256(path) != report_sha:
        raise ValueError("report changed during verification")
    result = dict(
        report_sha256=report_sha,
        verified_artifacts=len(seen_artifacts),
        verified_inputs=len(r["input_and_source_sha256"]),
        verification="hashes, all original depth points, pairing, fixed geometry across "
        "offsets, saved-map rescoring and aggregates; not stereo inference replay",
        common_effects=common_effects(common),
        aggregate=r["aggregate"],
        analyzer_sha256=file_sha256(__file__),
        registration_qualified=False,
        generator_training_approved=False,
        scope="same-sequence disjoint selection frames, previously observed in other "
        "diagnostics; LiDAR projection is not independent dense RGB-IR GT",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
