"""Top-three motion hypotheses ranked by symmetric ungated RoMa mask overlap."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.audit_roma_crop_support import covers_box_corners
from scripts.prepare_roma_motion_crops import MOTION, MOTION_SHA, crop_bounds, crop_identity
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_antiuav_temporal_motion import residual_motion
from scripts.verify_antiuav_tiled_matching import safe_artifact
from scripts.verify_minima_roma import verify

ROMA = Path("experiments/registration_minima_roma_gpu_01/report.json")


def candidates(score, mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 4]
    ids.sort(key=lambda i: (-float(score[labels == i].sum()), i))
    return [labels == i for i in ids[:3]]


def overlap(forward, source, target):
    y, x = np.nonzero(source)
    if not len(x):
        return 0.0
    indices = np.linspace(0, len(x) - 1, min(256, len(x)), dtype=int)
    points = forward(np.c_[x[indices], y[indices]])
    valid = np.isfinite(points).all(1)
    valid &= ((points >= 0) & (points <= np.array(target.shape[::-1]) - 1)).all(1)
    hits = np.zeros(len(points), dtype=bool)
    xy = np.floor(points[valid] + 0.5).astype(int)
    hits[valid] = target[xy[:, 1], xy[:, 0]]
    return float(hits.mean())


def select_pair(masks, predictors):
    choices = []
    for i, a in enumerate(masks[0]):
        for j, b in enumerate(masks[1]):
            f, r = overlap(predictors[0], a, b), overlap(predictors[1], b, a)
            choices.append(
                dict(
                    indices=[i, j],
                    directional_overlap=[f, r],
                    score=2 * f * r / (f + r) if f + r else 0.0,
                )
            )
    # Abstain when no reciprocal overlap exists; stable motion-rank tie breaking.
    best = max(choices, key=lambda c: c["score"], default=None)
    return (best if best is not None and best["score"] > 0 else None), choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(MOTION) != MOTION_SHA:
        raise ValueError("motion report changed")
    replay = verify(ROMA)
    motion, roma = [json.loads(p.read_text()) for p in (MOTION, ROMA)]
    hashes = motion["input_and_source_sha256"].copy()
    hashes.update(roma["input_and_source_sha256"])
    all_masks = {}
    for row in motion["rows"]:
        path = safe_artifact(MOTION.parent, row)
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            score, mask, threshold = residual_motion(data["flows"])
            np.testing.assert_array_equal(score, data["score"])
            np.testing.assert_array_equal(mask, data["mask"])
            if threshold != row["threshold"]:
                raise ValueError("motion replay differs")
            key = row["sequence_id"], row["decoded"][0]["frame"], row["modality"]
            if key in all_masks:
                raise ValueError("duplicate motion input")
            all_masks[key] = candidates(score, mask)
    rows = []
    for row in roma["rows"]:
        seq, frame = row["sequence_id"], row["frame_index"]
        masks = [all_masks[seq, frame, m] for m in ("visible", "infrared")]
        path = safe_artifact(ROMA.parent, row)
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            maps = split_dense(data["warp"], data["certainty"])
            sizes = [tuple(s[::-1]) for s in row["crop_shapes"]]
            predictors = [
                dense_predictor(f, np.ones_like(c), sizes[d], sizes[1 - d])
                for d, (f, c) in enumerate(maps)
            ]
            selected, choices = select_pair(masks, predictors)
        bounds = [[crop_bounds(m) for m in ms] for ms in masks]
        # Only here are annotations consulted, after candidate generation and ranking.
        support = [
            [covers_box_corners(b, row["boxes_xyxy"][d], row["header_rows"][d]) for b in bounds[d]]
            for d in range(2)
        ]
        selected_bounds = (
            [bounds[d][selected["indices"][d]] for d in range(2)] if selected else [None, None]
        )
        selected_support = (
            [support[d][selected["indices"][d]] for d in range(2)] if selected else [False, False]
        )
        ious = [
            corner_iou(
                crop_identity(
                    selected_bounds[d],
                    selected_bounds[1 - d],
                    row["header_rows"][d],
                    row["header_rows"][1 - d],
                ),
                row["boxes_xyxy"][d],
                row["boxes_xyxy"][1 - d],
            )
            for d in range(2)
        ]
        rows.append(
            dict(
                sequence_id=seq,
                frame_index=frame,
                candidates=bounds,
                overlap_scores=choices,
                selected=selected,
                selected_crop_bounds=selected_bounds,
                candidate_corner_support=support,
                selected_corner_support=selected_support,
                identity_iou=ious,
                oracle_candidate_coverage=all(any(s) for s in support),
            )
        )
    for p in (
        MOTION,
        ROMA,
        Path(__file__),
        Path("scripts/prepare_roma_motion_crops.py"),
        Path("scripts/audit_roma_crop_support.py"),
    ):
        hashes[str(p)] = file_sha256(p)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        primary_replay=replay,
        pairs=len(rows),
        oracle_candidate_coverage=sum(r["oracle_candidate_coverage"] for r in rows),
        selected_joint_corner_coverage=sum(all(r["selected_corner_support"]) for r in rows),
        abstentions=sum(r["selected"] is None for r in rows),
        identity_joint_box_passes=sum(
            all(v is not None and v >= 0.6 for v in r["identity_iou"]) for r in rows
        ),
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Top3 oracle coverage is a post-hoc upper bound, not deployable performance.",
            "Ungated RoMa overlap may connect incorrect regions; not pixel GT.",
            "Motion candidates include background; crop coverage is not correct registration.",
            "No new neural inference; existing dense fields only.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "input_and_source_sha256")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
