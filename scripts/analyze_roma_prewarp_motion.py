"""Attribute residual matcher changes without fitting or changing qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.prewarp import field_predictor, observed
from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.probe_roma_cycle_rejection import finite_predict
from scripts.probe_roma_prewarp_gpu import INPUT, protocol, scores
from scripts.verify_antiuav_tiled_matching import safe_artifact

REPORT = Path("experiments/registration_roma_prewarp_gpu_01/report.json")
SHA = "b33c04fd9eff28c53014dd194506b19dfb38a222214ded4bde3a9be06218e778"


def box_grid(box):
    lo, hi = np.asarray(box[:2]), np.asarray(box[2:])
    if np.any(hi <= lo):
        raise ValueError("positive box required")
    f = (np.arange(9) + 0.5) / 9
    x, y = np.meshgrid(lo[0] + f * (hi[0] - lo[0]), lo[1] + f * (hi[1] - lo[1]))
    return np.c_[x.ravel(), y.ravel()]


def summarize_motion(query, predicted, returned, diagonal, support):
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("positive finite diagonal required")
    motion = np.linalg.norm(predicted - query, axis=1)
    cycle = np.linalg.norm(returned - query, axis=1)
    valid = np.asarray(support, dtype=bool) & np.isfinite(motion) & np.isfinite(cycle)
    return dict(
        queries=len(query),
        observed_cycle_queries=int(valid.sum()),
        median_motion_pixels=float(np.median(motion[valid])) if valid.any() else None,
        median_motion_box_diagonals=float(np.median(motion[valid]) / diagonal)
        if valid.any()
        else None,
        cycle_le_one=int((valid & (cycle <= 1)).sum()),
        motion_gt_box_diagonal=int((valid & (motion > diagonal)).sum()),
        cycle_le_one_and_motion_gt_box_diagonal=int(
            (valid & (cycle <= 1) & (motion > diagonal)).sum()
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(REPORT) != SHA:
        raise ValueError("inference report changed")
    manifest, hashes = protocol()
    report = json.loads(REPORT.read_text())
    if report["input_and_source_sha256"] != hashes:
        raise ValueError("inference provenance differs")
    if [r["sequence_id"] for r in report["rows"]] != [r["sequence_id"] for r in manifest["rows"]]:
        raise ValueError("pair inventory differs")
    rows = []
    for row, original in zip(report["rows"], manifest["rows"], strict=True):
        artifact = safe_artifact(REPORT.parent, row)
        hashes[str(artifact)] = row["sha256"]
        with (
            np.load(artifact, allow_pickle=False) as saved,
            np.load(safe_artifact(INPUT.parent, original), allow_pickle=False) as prepared,
        ):
            if scores(saved["warp"], saved["certainty"], original, prepared) != row["scores"]:
                raise ValueError("original score differs")
            size = original["image_sizes"][1]
            top = original["header_rows"][1]
            crop_size = (size[0], size[1] - top)
            maps = split_dense(saved["warp"], saved["certainty"])
            predictors = [
                dense_predictor(f, np.ones_like(c), crop_size, crop_size, top, top) for f, c in maps
            ]
            coarse = field_predictor(prepared["rgb_to_ir_field"], original["image_sizes"][0], size)
            query = coarse(box_grid(original["boxes_xyxy"][0]))
            predicted = finite_predict(predictors[0], query)
            returned = finite_predict(predictors[1], predicted)
            box = np.asarray(original["boxes_xyxy"][1])
            summary = summarize_motion(
                query,
                predicted,
                returned,
                np.linalg.norm(box[2:] - box[:2]),
                observed(prepared["observed"], query),
            )
            initial = all(v is not None and v >= 0.6 for v in original["coarse_box_iou"])
            final = row["scores"]["ungated_diagnostic"]["both_ge_06"]
            rows.append(
                dict(
                    sequence_id=row["sequence_id"],
                    initial_box_pass=initial,
                    refined_box_pass=final,
                    motion=summary,
                )
            )
    for path in (REPORT, Path(__file__), Path("scripts/probe_roma_cycle_rejection.py")):
        hashes[str(path)] = file_sha256(path)
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        units="uncropped IR base-grid pixels",
        limitations=[
            "Post-hoc attribution, not a fitted correction.",
            "Boxes define diagnostic queries, not physical correspondences.",
            "Large residual motion is not itself proof of incorrect matching.",
            "Invalid cycles remain in the all-query denominator.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
