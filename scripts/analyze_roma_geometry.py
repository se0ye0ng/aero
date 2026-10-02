"""Audit ungated RoMa cycles and local orientation, not physical qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.verify_antiuav_tiled_matching import safe_artifact
from scripts.verify_minima_roma import verify


def box_grid(box):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
        raise ValueError("positive finite xyxy box required")
    # Interior samples of the annotation box; no assertion of foreground membership.
    fractions = (np.arange(9) + 0.5) / 9
    x, y = np.meshgrid(
        box[0] + fractions * (box[2] - box[0]), box[1] + fractions * (box[3] - box[1])
    )
    return np.c_[x.ravel(), y.ravel()]


def audit_queries(forward, reverse, query, target_box):
    query = np.asarray(query, dtype=float)
    target = forward(query)
    valid = np.isfinite(target).all(1)
    back = np.full_like(target, np.nan)
    if valid.any():
        back[valid] = reverse(target[valid])
    cycle = np.linalg.norm(back - query, axis=1)
    cycle_ok = np.isfinite(cycle) & (cycle <= 3)
    box = np.asarray(target_box)
    in_box = valid & (target >= box[:2]).all(1) & (target < box[2:]).all(1)
    # Symmetric one-source-pixel finite differences. Unsupported stencils stay invalid.
    dx = (forward(query + [1, 0]) - forward(query - [1, 0])) / 2
    dy = (forward(query + [0, 1]) - forward(query - [0, 1])) / 2
    determinant = dx[:, 0] * dy[:, 1] - dy[:, 0] * dx[:, 1]
    finite_det = determinant[np.isfinite(determinant)]
    finite_cycle = cycle[np.isfinite(cycle)]
    return dict(
        queries=len(query),
        forward_supported=int(valid.sum()),
        cycle_supported=len(finite_cycle),
        cycle_le_3_all_fraction=float(cycle_ok.mean()),
        cycle_median_supported_px=float(np.median(finite_cycle)) if len(finite_cycle) else None,
        cycle_p95_supported_px=float(np.percentile(finite_cycle, 95))
        if len(finite_cycle)
        else None,
        target_box_hits=int(in_box.sum()),
        cycle_le_3_but_outside_target_box=int((cycle_ok & ~in_box).sum()),
        jacobian_supported=len(finite_det),
        jacobian_nonpositive=int((finite_det <= 0).sum()),
        jacobian_nonpositive_supported_fraction=float(np.mean(finite_det <= 0))
        if len(finite_det)
        else None,
    )


def analyze(path):
    primary = verify(path)
    source = json.loads(path.read_text())
    rows = []
    for row in source["rows"]:
        with np.load(safe_artifact(path.parent, row), allow_pickle=False) as data:
            maps = split_dense(data["warp"], data["certainty"])
            sizes = [tuple(s[::-1]) for s in row["crop_shapes"]]
            predictors = [
                dense_predictor(
                    f,
                    np.ones_like(c),
                    sizes[d],
                    sizes[1 - d],
                    row["header_rows"][d],
                    row["header_rows"][1 - d],
                )
                for d, (f, c) in enumerate(maps)
            ]
            directions = []
            for d in range(2):
                w, h = sizes[d]
                yy, xx = np.mgrid[16 : h - 16 : 16, 16 : w - 16 : 16]
                frame = np.c_[xx.ravel(), yy.ravel() + row["header_rows"][d]]
                directions.append(
                    dict(
                        source="visible" if d == 0 else "infrared",
                        ungated_box_iou=corner_iou(
                            predictors[d], row["boxes_xyxy"][d], row["boxes_xyxy"][1 - d]
                        ),
                        roi=audit_queries(
                            predictors[d],
                            predictors[1 - d],
                            box_grid(row["boxes_xyxy"][d]),
                            row["boxes_xyxy"][1 - d],
                        ),
                        frame=audit_queries(
                            predictors[d], predictors[1 - d], frame, row["boxes_xyxy"][1 - d]
                        ),
                    )
                )
        rows.append(
            dict(
                sequence_id=row["sequence_id"],
                frame_index=row["frame_index"],
                directions=directions,
            )
        )
    return dict(
        primary_verification=primary,
        source_sha256=file_sha256(__file__),
        rows=rows,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Ungated diagnostic; original confidence gate is unchanged.",
            "GT boxes select diagnostic queries only, not model inference or fitting.",
            "Box interiors include background; containment is not pixel GT.",
            "Cycle consistency and sampled Jacobians do not prove correct physical correspondence.",
            "Cycle errors are base source pixels, not native video pixels.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = analyze(args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out}; diagnostic only, not qualification")


if __name__ == "__main__":
    main()
