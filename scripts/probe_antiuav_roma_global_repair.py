"""Evaluate fixed image-only affine fallback on Anti-UAV, using box proxies only."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_roma_global_repair import repair
from scripts.verify_antiuav_tiled_matching import safe_artifact
from scripts.verify_minima_roma import verify

REPORT = Path("experiments/registration_minima_roma_gpu_01/report.json")
SHA = "53f123f1e03ac563585cd477ae7b4748af875c830bed05ca9d8b8c0dbedc367d"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(REPORT) != SHA:
        raise ValueError("frozen report changed")
    verification = verify(REPORT)
    report = json.loads(REPORT.read_text())
    hashes = report["input_and_source_sha256"].copy()
    rows = []
    for row in report["rows"]:
        artifact = safe_artifact(REPORT.parent, row)
        hashes[str(artifact)] = row["sha256"]
        with np.load(artifact, allow_pickle=False) as saved:
            maps = split_dense(saved["warp"], saved["certainty"])
            sizes = [tuple(shape[::-1]) for shape in row["crop_shapes"]]
            predictors = [
                dense_predictor(f, np.ones_like(c), sizes[d], sizes[1 - d])
                for d, (f, c) in enumerate(maps)
            ]
            boxes = [
                np.asarray(b) - [0, top, 0, top]
                for b, top in zip(row["boxes_xyxy"], row["header_rows"], strict=True)
            ]
            directions = []
            for d in range(2):
                x0, y0, x1, y1 = boxes[d]
                queries = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                before, after, affine, changed, fit = repair(
                    predictors[d], predictors[1 - d], queries, sizes[d], sizes[1 - d]
                )
                scores = {}
                for name, prediction in (
                    ("original", before),
                    ("fallback", after),
                    ("affine_only", affine),
                ):
                    scores[name] = corner_iou(
                        lambda q, prediction=prediction: prediction, boxes[d], boxes[1 - d]
                    )
                directions.append(dict(scores=scores, changed=int(changed.sum()), fit=fit))
        rows.append(
            dict(
                sequence_id=row["sequence_id"],
                frame_index=row["frame_index"],
                directions=directions,
            )
        )
    counts = {
        name: sum(
            all(
                d["scores"][name] is not None and d["scores"][name] >= 0.6
                for d in row["directions"]
            )
            for row in rows
        )
        for name in ("original", "fallback", "affine_only")
    }
    for path in (
        REPORT,
        Path(__file__),
        Path("scripts/probe_roma_global_repair.py"),
        Path("scripts/probe_roma_cycle_rejection.py"),
        Path("scripts/probe_roma_local_repair.py"),
    ):
        hashes[str(path)] = file_sha256(path)
    result = dict(
        rows=rows,
        both_direction_box_iou_ge_06=counts,
        primary_verification=verification,
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Box corners are queries only, never fitting controls.",
            "Box overlap is not independent pixel correspondence GT.",
            "Exploratory reuse of fixed train16, not held-out qualification.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
