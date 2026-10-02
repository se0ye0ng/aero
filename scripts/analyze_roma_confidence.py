"""Separate fixed confidence rejection from geometry; never change qualification."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.verify_antiuav_tiled_matching import safe_artifact
from scripts.verify_minima_roma import verify


def summarize_control(errors, confidence):
    errors, confidence = np.asarray(errors), np.asarray(confidence)
    if errors.ndim != 1 or errors.shape != confidence.shape or not len(errors):
        raise ValueError("nonempty equal-length vectors required")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("confidence must lie in [0, 1]")
    bins = []
    for low, high in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)):
        selected = (confidence >= low) & (confidence < high)
        n = int(selected.sum())
        correct = int((selected & (errors <= 3) & np.isfinite(errors)).sum())
        bins.append(
            dict(
                low=low,
                high_exclusive=high,
                queries=n,
                correct=correct,
                pck3_all=correct / n if n else None,
            )
        )
    finite = errors[np.isfinite(errors)]
    return dict(
        queries=len(errors),
        supported=len(finite),
        bins=bins,
        pck3_all=float(np.mean(np.isfinite(errors) & (errors <= 3))),
        median_supported_error_px=float(np.median(finite)) if len(finite) else None,
    )


def analyze(path):
    primary = verify(path)
    report = json.loads(path.read_text())
    rows = []
    for row in report["rows"]:
        with np.load(safe_artifact(path.parent, row), allow_pickle=False) as data:
            maps = split_dense(data["warp"], data["certainty"])
            sizes = [tuple(shape[::-1]) for shape in row["crop_shapes"]]
            ious = []
            for d, (field, confidence) in enumerate(maps):
                # Disable only confidence filtering. In-frame support remains mandatory.
                predictor = dense_predictor(
                    field,
                    np.ones_like(confidence),
                    sizes[d],
                    sizes[1 - d],
                    row["header_rows"][d],
                    row["header_rows"][1 - d],
                )
                ious.append(corner_iou(predictor, row["boxes_xyxy"][d], row["boxes_xyxy"][1 - d]))
        rows.append(
            dict(
                sequence_id=row["sequence_id"],
                frame_index=row["frame_index"],
                primary_iou=row["scores"]["iou"],
                ungated_iou=ious,
            )
        )
    control = next(c for c in report["controls"] if c["kind"] == "known_shift")
    # The control uses the first frozen case, independently of report row order.
    from scripts.probe_minima_roma_gpu import protocol

    _, cases, metadata = protocol()
    first = cases[0]
    h, w = metadata[first["sequence_id"]]["visible"]["resized_shape"][:2]
    h -= first["header_rows"][0]
    yy, xx = np.mgrid[16 : h - 16 : 16, 16 : w - 16 : 16]
    query = np.c_[xx.ravel(), yy.ravel()]
    with np.load(safe_artifact(path.parent, control), allow_pickle=False) as data:
        field, confidence = split_dense(data["warp"], data["certainty"])[0]
        pred = dense_predictor(field, np.ones_like(confidence), (w, h), (w, h))(query)
        error = np.linalg.norm(pred - query - [8, -8], axis=1)
        grid = (query + 0.5) / [w, h] * [confidence.shape[1], confidence.shape[0]] - 0.5
        sampled = map_coordinates(
            confidence, grid[:, ::-1].T, order=1, mode="nearest", prefilter=False
        )
    return dict(
        primary_verification=primary,
        rows=rows,
        ungated_box_proxy_both_ge_06=sum(
            all(v is not None and v >= 0.6 for v in r["ungated_iou"]) for r in rows
        ),
        known_shift_ungated=summarize_control(error, sampled),
        source_sha256=file_sha256(__file__),
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Post-hoc diagnostic, not a new passing threshold.",
            "Same-RGB synthetic control does not establish RGB-IR accuracy.",
            "Box envelope overlap is not independent pixel correspondence GT.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = analyze(args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
