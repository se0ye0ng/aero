"""Fit bounded geometric alternatives to saved XoFTR matches, not correspondence GT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from aero_ir.registration.detector_free import reciprocal_mask
from aero_ir.utils.manifest import file_sha256

MINIMUM_PAIRS = {"similarity": 6, "affine": 8, "homography": 12}
CONDITIONS = ("raw_postfilter", "input_header_crop")


def corners(box):
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def project(matrix, points):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    if not np.isfinite(homogeneous).all() or (np.abs(homogeneous[:, 2]) < 1e-9).any():
        raise ValueError("nonfinite projection or pole")
    return homogeneous[:, :2] / homogeneous[:, 2:]


def domain_has_pole(matrix, box):
    denominators = (np.c_[corners(box), np.ones(4)] @ matrix.T)[:, 2]
    return not (
        np.isfinite(denominators).all()
        and ((denominators > 1e-9).all() or (denominators < -1e-9).all())
    )


def box_iou(matrix, source_box, target_box):
    if domain_has_pole(matrix, source_box):
        return None
    warped = project(matrix, corners(source_box))
    low, high = warped.min(axis=0), warped.max(axis=0)
    overlap = np.maximum(
        np.minimum(high, target_box[2:]) - np.maximum(low, target_box[:2]), 0
    ).prod()
    union = np.prod(high - low) + np.prod(target_box[2:] - target_box[:2]) - overlap
    return float(overlap / union) if union > 0 else None


def geometric_coverage(matrix, domain0, domain1):
    """32x32 retained-domain centre lattice; numerical support, not physical coverage."""
    fractions = (np.arange(32) + 0.5) / 32
    xx, yy = np.meshgrid(
        domain0[0] + fractions * (domain0[2] - domain0[0]),
        domain0[1] + fractions * (domain0[3] - domain0[1]),
    )
    mapped = project(matrix, np.c_[xx.ravel(), yy.ravel()])
    return float(((mapped >= domain1[:2]).all(1) & (mapped <= domain1[2:]).all(1)).mean())


def has_area(points):
    return len(points) >= 3 and np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-5) == 2


def unique_reciprocal(a, b, reverse0, reverse1):
    arrays = [np.asarray(p, dtype=np.float64) for p in (a, b, reverse0, reverse1)]
    if any(p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() for p in arrays):
        raise ValueError("matches must be finite Nx2 arrays")
    a, b, reverse0, reverse1 = arrays
    if len(a) != len(b) or len(reverse0) != len(reverse1):
        raise ValueError("paired lengths differ")
    keep = reciprocal_mask(a, b, reverse0, reverse1, tolerance=2.0)
    pairs = np.c_[a[keep], b[keep]]
    # Round ONLY for deduplication; fit the original subpixel coordinates.
    _, indices = np.unique(np.round(pairs), axis=0, return_index=True)
    pairs = pairs[indices]
    return pairs[:, :2], pairs[:, 2:]


def fit_geometry(a, b, model, boxes, domains):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    result = {"model": model, "unique_reciprocal_pairs": len(a)}
    if model not in MINIMUM_PAIRS:
        raise ValueError("unknown geometric model")
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("expected paired Nx2 arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {**result, "status": "nonfinite_matches"}
    if len(a) < MINIMUM_PAIRS[model]:
        return {**result, "status": "insufficient_matches"}
    order = np.random.default_rng(0).permutation(len(a))
    fit, check = np.array_split(order, 2)
    result.update(fit_count=len(fit), check_count=len(check))
    if not has_area(a[fit]) or not has_area(b[fit]):
        return {**result, "status": "degenerate_fit_support"}
    cv2.setRNGSeed(0)
    if model == "homography":
        matrix, inliers = cv2.findHomography(
            a[fit], b[fit], cv2.RANSAC, 3.0, maxIters=5000, confidence=0.999
        )
    else:
        estimator = cv2.estimateAffinePartial2D if model == "similarity" else cv2.estimateAffine2D
        matrix, inliers = estimator(
            a[fit],
            b[fit],
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.999,
            refineIters=10,
        )
        if matrix is not None:
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return {**result, "status": "fit_failed"}
    inliers = inliers.ravel().astype(bool)
    required = 4 if model == "homography" else 3
    if inliers.sum() < required or not has_area(a[fit][inliers]) or not has_area(b[fit][inliers]):
        return {**result, "status": "degenerate_inlier_support"}
    if np.linalg.cond(matrix) > 1e12:
        return {**result, "status": "singular_or_ill_conditioned"}
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return {**result, "status": "singular_or_ill_conditioned"}
    if any(
        domain_has_pole(h, domain)
        for h, domain in (
            (matrix, domains[0]),
            (inverse, domains[1]),
            (matrix, boxes[0]),
            (inverse, boxes[1]),
        )
    ):
        return {**result, "status": "pole_crosses_domain_or_box"}
    forward = np.linalg.norm(project(matrix, a[check]) - b[check], axis=1)
    reverse = np.linalg.norm(project(inverse, b[check]) - a[check], axis=1)
    result.update(
        status="fit",
        matrix=matrix.tolist(),
        inverse_matrix=inverse.tolist(),
        fit_inlier_fraction=float(inliers.mean()),
        forward_box_iou_proxy=box_iou(matrix, *boxes),
        inverse_box_iou_proxy=box_iou(inverse, *boxes[::-1]),
        forward_retained_domain_support=geometric_coverage(matrix, *domains),
        inverse_retained_domain_support=geometric_coverage(inverse, *domains[::-1]),
        heldout_consistency_not_accuracy={
            name: {
                "median_pixels": float(np.median(errors)),
                "p95_pixels": float(np.quantile(errors, 0.95)),
                "fraction_within_3px": float((errors <= 3).mean()),
            }
            for name, errors in (("rgb_to_ir", forward), ("ir_to_rgb", reverse))
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("experiments/xoftr_overlay_train16_01/report.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite geometry probe")
    source = json.loads(args.report.read_text())
    if (
        source.get("evaluated_split") != "train"
        or source.get("validation_or_test_access") != "none"
    ):
        raise ValueError("train-only source required")
    inputs = {(p["sequence_id"], p["frame_index"]): p for p in source["inputs"]}
    rows, condition_keys, provenance = [], {}, {}
    for condition in CONDITIONS:
        selected = [r for r in source["rows"] if r["condition"] == condition]
        keys = [(r["sequence_id"], r["frame_index"]) for r in selected]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate pairs")
        condition_keys[condition] = set(keys)
        for row in selected:
            key = (row["sequence_id"], row["frame_index"])
            path = (args.report.parent / row["matches_file"]).resolve()
            if not path.is_relative_to(args.report.parent.resolve()):
                raise ValueError("matches path outside source directory")
            if file_sha256(path) != row["matches_sha256"]:
                raise ValueError(f"matches hash mismatch: {path}")
            provenance[row["matches_file"]] = row["matches_sha256"]
            with np.load(path, allow_pickle=False) as saved:
                a, b = unique_reciprocal(
                    *(saved[k] for k in ("points0", "points1", "reverse0", "reverse1"))
                )
            domains = []
            for i, m in enumerate(("visible", "infrared")):
                h, w = inputs[key]["inputs"][m]["resized_shape"]
                domains.append(np.array([0, row["header_rows"][i], w - 1, h - 1], dtype=float))
            boxes = [np.array(box, dtype=float) for box in row["boxes_xyxy"]]
            for model in MINIMUM_PAIRS:
                result = fit_geometry(a, b, model, boxes, domains)
                rows.append(
                    {
                        "sequence_id": key[0],
                        "frame_index": key[1],
                        "condition": condition,
                        "target_fully_retained": row["target_fully_retained"],
                        **result,
                    }
                )
    if condition_keys[CONDITIONS[0]] != condition_keys[CONDITIONS[1]] or not rows:
        raise ValueError("conditions must cover identical nonempty frame panel")
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for model in MINIMUM_PAIRS:
            selected = [r for r in rows if r["condition"] == condition and r["model"] == model]
            valid = [r for r in selected if r["status"] == "fit"]
            summary[condition][model] = {
                "frames": len(selected),
                "eligible_fits": len(valid),
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "both_box_iou_ge_0_6_all_frames": sum(
                    min(r["forward_box_iou_proxy"], r["inverse_box_iou_proxy"]) >= 0.6
                    for r in valid
                ),
                **{
                    f"mean_{metric}": float(np.mean([r[metric] for r in valid])) if valid else None
                    for metric in (
                        "forward_box_iou_proxy",
                        "inverse_box_iou_proxy",
                        "forward_retained_domain_support",
                        "inverse_retained_domain_support",
                    )
                },
                "median_heldout_forward_fraction_within_3px": float(
                    np.median(
                        [
                            r["heldout_consistency_not_accuracy"]["rgb_to_ir"][
                                "fraction_within_3px"
                            ]
                            for r in valid
                        ]
                    )
                )
                if valid
                else None,
            }
    result = {
        "kind": "saved_match_geometry_diagnostic",
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "source_report_sha256": file_sha256(args.report),
        "source_hashes": {str(Path(__file__)): file_sha256(__file__)},
        "verified_matches_sha256": provenance,
        "protocol": {
            "reciprocal_tolerance_px": 2,
            "dedup": "round joint pairs to 1px",
            "minimum_total_pairs": MINIMUM_PAIRS,
            "ransac_threshold_px": 3,
            "fit_check_split": "fixed RNG0 halves; no refit on check",
            "boxes_used_for_fitting": False,
            "identity_fallback": False,
            "coverage": "32x32 retained-domain lattice, numerical support not physical GT",
            "inverse": "analytic inverse of fitted forward transform, not independent evidence",
        },
        "rows": rows,
        "summary": summary,
        "generator_training_eligible": "hold",
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "report.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
