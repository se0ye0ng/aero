"""Freeze annotation-free motion crops and a crop-coordinate identity control."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_antiuav_temporal_motion import residual_motion
from scripts.probe_minima_roma_gpu import protocol
from scripts.verify_antiuav_tiled_matching import safe_artifact

MOTION = Path("experiments/registration_antiuav_temporal_motion_01/report.json")
MOTION_SHA = "aa22edda051d70c1cf9a0978cb290bdbd366f9265c84607526c0a266e092f0c6"


def select_component(score, mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    candidates = [j for j in range(1, count) if stats[j, cv2.CC_STAT_AREA] >= 4]
    best = max(candidates, key=lambda j: float(score[labels == j].sum()), default=0)
    return labels == best if best else np.zeros_like(mask)


def crop_bounds(component):
    """Three times the component bounding-box long side; clip to actual image."""
    mask = np.asarray(component)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("2D boolean component required")
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    lo = np.array([x.min(), y.min()])
    hi = np.array([x.max() + 1, y.max() + 1])
    centre = (lo + hi) / 2
    side = 3 * max(hi - lo)
    start = np.maximum(0, np.floor(centre - side / 2)).astype(int)
    end = np.minimum(mask.shape[::-1], np.ceil(centre + side / 2)).astype(int)
    return [int(v) for v in np.r_[start, end]]


def crop_identity(source, target, source_top=0, target_top=0):
    def predict(points):
        points = np.asarray(points, dtype=float)
        if source is None or target is None:
            return np.full_like(points, np.nan)
        a, b = np.asarray(source), np.asarray(target)
        local = points - [a[0], a[1] + source_top]
        valid = ((local >= 0) & (local <= a[2:] - a[:2] - 1)).all(1)
        out = (local + 0.5) / (a[2:] - a[:2]) * (b[2:] - b[:2]) - 0.5
        valid &= ((out >= 0) & (out <= b[2:] - b[:2] - 1)).all(1)
        out += [b[0], b[1] + target_top]
        out[~valid] = np.nan
        return out

    return predict


def prepare():
    hashes, cases, _ = protocol()
    if file_sha256(MOTION) != MOTION_SHA:
        raise ValueError("motion evidence changed")
    report = json.loads(MOTION.read_text())
    for path, digest in report["input_and_source_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"motion source changed: {path}")
        hashes[path] = digest
    crops = {}
    for row in report["rows"]:
        path = safe_artifact(MOTION.parent, row)
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            score, mask, threshold = residual_motion(data["flows"])
            np.testing.assert_array_equal(score, data["score"])
            np.testing.assert_array_equal(mask, data["mask"])
            if threshold != row["threshold"]:
                raise ValueError("motion threshold changed")
            component = select_component(score, mask)
            np.testing.assert_array_equal(component, data["selected_component"])
            crop = crop_bounds(component)
        key = row["sequence_id"], row["decoded"][0]["frame"], row["modality"]
        if key in crops:
            raise ValueError("duplicate motion row")
        crops[key] = crop
    expected = {
        (c["sequence_id"], c["frame_index"], m) for c in cases for m in ("visible", "infrared")
    }
    if set(crops) != expected:
        raise ValueError("motion inventory differs")
    rows = []
    for case in cases:
        seq, frame = case["sequence_id"], case["frame_index"]
        bounds = [crops[seq, frame, m] for m in ("visible", "infrared")]
        tops, boxes = case["header_rows"], case["boxes_xyxy"]
        ious = [
            corner_iou(
                crop_identity(bounds[d], bounds[1 - d], tops[d], tops[1 - d]),
                boxes[d],
                boxes[1 - d],
            )
            for d in range(2)
        ]
        rows.append(
            dict(
                sequence_id=seq,
                frame_index=frame,
                crop_bounds=bounds,
                header_rows=tops,
                boxes_xyxy=boxes,
                identity_iou=ious,
                identity_box_proxy_both_ge_06=all(v is not None and v >= 0.6 for v in ious),
            )
        )
    hashes[str(MOTION)] = MOTION_SHA
    hashes[str(Path(__file__))] = file_sha256(__file__)
    return dict(
        rows=rows,
        input_and_source_sha256=hashes,
        crop_policy="largest integrated residual-motion component, area>=4; square 3x long side",
        gt_use="post-hoc scoring only; no crop selection or fitting",
        identity_box_proxy_both_ge_06=sum(r["identity_box_proxy_both_ge_06"] for r in rows),
        registration_qualified=False,
        generator_training_approved=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = prepare()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                pairs=len(result["rows"]),
                identity_box_proxy_both_ge_06=result["identity_box_proxy_both_ge_06"],
            )
        )
    )


if __name__ == "__main__":
    main()
