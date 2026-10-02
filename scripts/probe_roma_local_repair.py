"""Exploratory image-map-only local affine repair; never qualification approval."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_roma import inputs, score
from scripts.probe_roma_cycle_rejection import DATA, REPORT, REPORT_SHA, admission
from scripts.verify_antiuav_tiled_matching import safe_artifact


def local_prediction(source, target, queries, radius=0.15):
    """Normalized coordinates; require 12 neighbors and interpolation, not extrapolation."""
    out = np.full_like(queries, np.nan, dtype=float)
    if len(source) < 12:
        return out
    tree = cKDTree(source)
    distances, indices = tree.query(queries, k=min(32, len(source)))
    for i, (distance, index) in enumerate(zip(distances, indices, strict=True)):
        index = index[distance <= radius]
        if len(index) < 12:
            continue
        x = source[index] - queries[i]
        design = np.c_[x, np.ones(len(x))]
        if np.linalg.matrix_rank(design) < 3:
            continue
        if Delaunay(x).find_simplex(np.zeros((1, 2)))[0] < 0:
            continue
        weights = 1 / np.maximum(np.linalg.norm(x, axis=1), 0.01)
        coef = np.linalg.lstsq(
            design * weights[:, None], target[index] * weights[:, None], rcond=None
        )[0]
        if np.linalg.det(coef[:2]) <= 0:
            continue
        out[i] = coef[2]
    return out


def repair(forward, reverse, queries, source_size, target_size):
    # Fixed image-wide grid, independent of landmarks and target boxes.
    xx, yy = np.meshgrid((np.arange(64) + 0.5) / 64, (np.arange(64) + 0.5) / 64)
    grid = np.c_[xx.ravel(), yy.ravel()] * source_size - 0.5
    values, trusted, _ = admission(forward, reverse, grid, source_size, target_size)
    original, accepted, _ = admission(forward, reverse, queries, source_size, target_size)
    proposed = (
        local_prediction(
            (grid[trusted] + 0.5) / source_size,
            (values[trusted] + 0.5) / target_size,
            (queries + 0.5) / source_size,
        )
        * target_size
        - 0.5
    )
    available = np.isfinite(proposed).all(1)
    available &= ((proposed >= 0) & (proposed <= np.asarray(target_size) - 1)).all(1)
    changed = ~accepted & available
    output = original.copy()
    output[changed] = proposed[changed]
    return original, output, changed, int(trusted.sum())


def metrics(pred, refs):
    error = np.linalg.norm(pred - refs, axis=1)
    finite = np.isfinite(error)
    return dict(pck3_all=float((finite & (error <= 3)).mean()), unavailable=int((~finite).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(REPORT) != REPORT_SHA:
        raise ValueError("inference report changed")
    report = json.loads(REPORT.read_text())
    hashes = report["input_and_source_sha256"].copy()
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    rows = []
    for row in report["rows"]:
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
            before, after, changed, controls = repair(*predictors, refs[0], *sizes)
        rows.append(
            dict(
                pair=row["pair"],
                landmarks=len(refs[0]),
                controls=controls,
                changed=int(changed.sum()),
                before=metrics(before, refs[1]),
                after=metrics(after, refs[1]),
                repaired_predictions=[
                    [float(v) if np.isfinite(v) else None for v in p] for p in after
                ],
            )
        )
    for path in (REPORT, Path(__file__), Path("scripts/probe_roma_cycle_rejection.py")):
        hashes[str(path)] = file_sha256(path)
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        protocol=(
            "64x64 grid; cycle<=1 target-equivalent pixel; positive Jacobians; "
            "local affine 32 neighbors radius .15; min12; inside local convex hull; "
            "replace rejected queries only"
        ),
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Post-hoc same-scene external diagnostic, not Anti-UAV qualification.",
            "Cycle-consistent incorrect controls may contaminate the fit.",
            "Reference destinations used only in scoring; no GT control fitting.",
        ],
        duplicate_labels=report["duplicate_labels"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            [{k: v for k, v in row.items() if k != "repaired_predictions"} for row in rows],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
