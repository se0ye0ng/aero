"""Image-only robust affine fallback on the frozen external panel; diagnostic only."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_roma import inputs, score
from scripts.probe_roma_cycle_rejection import DATA, REPORT, REPORT_SHA, admission
from scripts.probe_roma_local_repair import metrics
from scripts.verify_antiuav_tiled_matching import safe_artifact


def robust_affine(source, target, query):
    """All units thermal-equivalent pixels; no reference destination input."""
    info = dict(controls=len(source), inliers=0, accepted=False)
    empty = np.full_like(query, np.nan, dtype=float)
    if len(source) < 32:
        return empty, info
    cv2.setRNGSeed(0)
    matrix, mask = cv2.estimateAffine2D(
        np.asarray(source, dtype=float),
        np.asarray(target, dtype=float),
        method=cv2.RANSAC,
        ransacReprojThreshold=3,
        maxIters=10000,
        confidence=0.999,
        refineIters=10,
    )
    if matrix is None or mask is None:
        return empty, info
    info["inliers"] = int(mask.sum())
    if (
        not np.isfinite(matrix).all()
        or np.linalg.det(matrix[:, :2]) <= 0
        or info["inliers"] < max(12, 0.5 * len(source))
    ):
        return empty, info
    info.update(accepted=True, matrix=matrix.tolist())
    return np.c_[query, np.ones(len(query))] @ matrix.T, info


def repair(forward, reverse, queries, source_size, target_size):
    x, y = np.meshgrid((np.arange(64) + 0.5) / 64, (np.arange(64) + 0.5) / 64)
    grid = np.c_[x.ravel(), y.ravel()] * source_size - 0.5
    values, trusted, _ = admission(forward, reverse, grid, source_size, target_size)
    original, accepted, _ = admission(forward, reverse, queries, source_size, target_size)
    scaling = np.asarray(target_size) / source_size
    proposed, fit = robust_affine(grid[trusted] * scaling, values[trusted], queries * scaling)
    available = np.isfinite(proposed).all(1)
    available &= ((proposed >= 0) & (proposed <= np.asarray(target_size) - 1)).all(1)
    proposed[~available] = np.nan
    changed = ~accepted & available
    fallback = original.copy()
    fallback[changed] = proposed[changed]
    return original, fallback, proposed, changed, fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(REPORT) != REPORT_SHA:
        raise ValueError("inference report changed")
    r = json.loads(REPORT.read_text())
    hashes = r["input_and_source_sha256"].copy()
    for path, sha in hashes.items():
        if file_sha256(path) != sha:
            raise ValueError(f"changed source: {path}")
    rows = []
    cv2.setNumThreads(1)
    for row in r["rows"]:
        images, refs = inputs(DATA, row["pair"])
        sizes = [im.size for im in images]
        artifact = safe_artifact(REPORT.parent, row)
        hashes[str(artifact)] = row["sha256"]
        with np.load(artifact, allow_pickle=False) as saved:
            if score(saved["warp"], saved["certainty"], sizes, refs) != row["scores"]:
                raise ValueError("baseline reconstruction differs")
            maps = split_dense(saved["warp"], saved["certainty"])
            predictors = [
                dense_predictor(f, np.ones_like(c), sizes[d], sizes[1 - d])
                for d, (f, c) in enumerate(maps)
            ]
            before, after, affine, changed, fit = repair(*predictors, refs[0], *sizes)
        rows.append(
            dict(
                pair=row["pair"],
                landmarks=len(refs[0]),
                fit=fit,
                changed=int(changed.sum()),
                original=metrics(before, refs[1]),
                fallback=metrics(after, refs[1]),
                affine_only=metrics(affine, refs[1]),
            )
        )
    for path in (
        REPORT,
        Path(__file__),
        Path("scripts/probe_roma_cycle_rejection.py"),
        Path("scripts/probe_roma_local_repair.py"),
    ):
        hashes[str(path)] = file_sha256(path)
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        duplicate_labels=r["duplicate_labels"],
        opencv_version=cv2.__version__,
        registration_qualified=False,
        generator_training_approved=False,
        protocol=(
            "64x64 cycle/orientation-selected controls; affine RANSAC3px; "
            "min32 controls; min50% inliers; replace rejected queries only"
        ),
        limitations=[
            "Post-hoc external diagnostic, not Anti-UAV qualification.",
            "Global affine extrapolation cannot establish local correctness.",
            "All configurations reported; no GT-based model selection.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
