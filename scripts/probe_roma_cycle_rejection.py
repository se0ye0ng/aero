"""Post-hoc cycle/orientation rejection diagnostic; not held-out qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_roma import inputs, score
from scripts.verify_antiuav_tiled_matching import safe_artifact

REPORT = Path("experiments/registration_roma_fresh_confirmation_gpu_01/report.json")
REPORT_SHA = "808564329bd00e6684d5cd418713e5b64a242e9875693e0fa7b5bfb71f868739"
DATA = Path("experiments/external/uav_tirvis_b6b91412_fresh8")


def finite_predict(predict, points):
    output = np.full_like(points, np.nan, dtype=float)
    valid = np.isfinite(points).all(1)
    if valid.any():
        output[valid] = predict(points[valid])
    return output


def determinant(predict, points):
    dx = (finite_predict(predict, points + [1, 0]) - finite_predict(predict, points - [1, 0])) / 2
    dy = (finite_predict(predict, points + [0, 1]) - finite_predict(predict, points - [0, 1])) / 2
    return dx[:, 0] * dy[:, 1] - dy[:, 0] * dx[:, 1]


def admission(forward, reverse, queries, source_size, target_size):
    """Image-map-only rule; no reference target coordinates are accepted."""
    queries = np.asarray(queries, dtype=float)
    predicted = forward(queries)
    returned = finite_predict(reverse, predicted)
    # Express source roundtrip displacement in target-equivalent pixel units.
    cycle = np.linalg.norm((returned - queries) / source_size * target_size, axis=1)
    det_forward = determinant(forward, queries)
    det_reverse = determinant(reverse, predicted)
    accepted = (
        np.isfinite(cycle)
        & (cycle <= 1.0)
        & np.isfinite(det_forward)
        & np.isfinite(det_reverse)
        & (det_forward > 0)
        & (det_reverse > 0)
    )
    return predicted, accepted, cycle


def summarize(errors, accepted):
    errors, accepted = np.asarray(errors), np.asarray(accepted, dtype=bool)
    good = np.isfinite(errors) & (errors <= 3)
    bad = ~good
    return dict(
        landmarks=len(errors),
        retained=int(accepted.sum()),
        retained_fraction=float(accepted.mean()),
        original_pck3_all=float(good.mean()),
        retained_pck3_all=float((good & accepted).mean()),
        conditional_retained_pck3=float(good[accepted].mean()) if accepted.any() else None,
        bad_points=int(bad.sum()),
        rejected_bad_points=int((bad & ~accepted).sum()),
        retained_bad_points=int((bad & accepted).sum()),
        rejected_good_points=int((good & ~accepted).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(REPORT) != REPORT_SHA:
        raise ValueError("frozen inference report changed")
    report = json.loads(REPORT.read_text())
    hashes = report["input_and_source_sha256"].copy()
    for path, expected in hashes.items():
        if file_sha256(path) != expected:
            raise ValueError(f"input changed: {path}")
    rows = []
    for row in report["rows"]:
        images, refs = inputs(DATA, row["pair"])
        sizes = [im.size for im in images]
        artifact = safe_artifact(REPORT.parent, row)
        hashes[str(artifact)] = row["sha256"]
        with np.load(artifact, allow_pickle=False) as data:
            if score(data["warp"], data["certainty"], sizes, refs) != row["scores"]:
                raise ValueError("original score differs")
            maps = split_dense(data["warp"], data["certainty"])
            predictors = [
                dense_predictor(f, np.ones_like(c), sizes[d], sizes[1 - d])
                for d, (f, c) in enumerate(maps)
            ]
            pred, accepted, cycle = admission(*predictors, refs[0], *sizes)
            errors = np.linalg.norm(pred - refs[1], axis=1)
        rows.append(
            dict(
                pair=row["pair"],
                summary=summarize(errors, accepted),
                accepted=accepted.tolist(),
                cycle_thermal_equivalent_px=[float(v) if np.isfinite(v) else None for v in cycle],
            )
        )
    hashes[str(REPORT)] = REPORT_SHA
    hashes[str(Path(__file__))] = file_sha256(__file__)
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        duplicate_labels=report["duplicate_labels"],
        selection_rule="cycle<=1 target-equivalent px; positive forward and reverse Jacobians",
        limitations=[
            "Exploratory reuse after viewing errors; not fresh confirmation.",
            "Selected-point accuracy must be accompanied by retained coverage.",
            "Perfectly invertible wrong correspondence can still pass this rule.",
            "Evaluated at authored landmarks, not a certified dense mask.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps([dict(pair=r["pair"], **r["summary"]) for r in rows], indent=2))


if __name__ == "__main__":
    main()
