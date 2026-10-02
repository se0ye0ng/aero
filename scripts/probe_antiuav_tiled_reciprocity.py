"""Fixed 3-pixel reciprocal endpoint filter on saved train16 matches, CPU only."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_external_local_warp import fit_image_warp
from scripts.verify_antiuav_tiled_matching import safe_artifact, verify


def reciprocal_ids(p, q, reverse_p, reverse_q, radius=3.0):
    """Mutual nearest endpoint pairs in 4D; both 2D distances must be <= radius.

    Reverse endpoints are swapped into forward coordinates. No boxes, confidence
    tuning, synthetic averaging, or independent nearest-endpoint joins are used.
    This is repeatability evidence, not a guarantee of correct correspondence.
    """
    arrays = [np.asarray(x, dtype=float) for x in (p, q, reverse_p, reverse_q)]
    if any(x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("finite Nx2 endpoints required")
    p, q, reverse_p, reverse_q = arrays
    if p.shape != q.shape or reverse_p.shape != reverse_q.shape or not 0 < radius < np.inf:
        raise ValueError("paired shapes and positive finite radius required")
    if not len(p) or not len(reverse_p):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    forward = np.c_[p, q]
    reverse = np.c_[reverse_q, reverse_p]
    _, nearest_r = cKDTree(reverse).query(forward, workers=1)
    _, nearest_f = cKDTree(forward).query(reverse, workers=1)
    good = nearest_f[nearest_r] == np.arange(len(p))
    good &= np.linalg.norm(p - reverse_q[nearest_r], axis=1) <= radius
    good &= np.linalg.norm(q - reverse_p[nearest_r], axis=1) <= radius
    ids = np.flatnonzero(good)
    return ids, nearest_r[ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    replay = verify(args.report)
    source = json.loads(args.report.read_text())
    settings_path = Path("configs/experiment/registration_external_local_warp_cpu.yaml")
    settings = yaml.safe_load(settings_path.read_text())
    files = [
        args.report,
        settings_path,
        Path(__file__),
        Path("scripts/verify_antiuav_tiled_matching.py"),
        Path("scripts/probe_external_local_warp.py"),
        Path("scripts/probe_antiuav_local_warp.py"),
    ]
    hashes = {str(p): file_sha256(p) for p in files}
    rows = []
    for row in source["rows"]:
        variants = {}
        path = safe_artifact(args.report.parent, row)
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as a:
            for variant, prefix in (("baseline", "baseline_"), ("tiled", "")):
                p, q, rp, rq = [
                    a[prefix + n] for n in ("points0", "points1", "reverse0", "reverse1")
                ]
                ids, reverse_ids = reciprocal_ids(p, q, rp, rq)
                fits = []
                # Fit before reading boxes: annotations cannot influence filtering.
                for x, y, c, keep, sizes in (
                    (p, q, a[prefix + "confidence"], ids, row["sizes"]),
                    (rp, rq, a[prefix + "reverse_confidence"], reverse_ids, row["sizes"][::-1]),
                ):
                    fits.append(fit_image_warp(x[keep], y[keep], c[keep], *sizes, settings))
                boxes = row["boxes_xyxy"]
                ious = [corner_iou(fits[0][0], *boxes), corner_iou(fits[1][0], *boxes[::-1])]
                variants[variant] = dict(
                    retained=len(ids),
                    original_counts=[len(p), len(rp)],
                    forward_ids=ids.tolist(),
                    reverse_ids=reverse_ids.tolist(),
                    fits=[f[1] for f in fits],
                    iou=ious,
                    joint_pass=all(v is not None and v >= 0.6 for v in ious),
                )
        rows.append(
            dict(sequence_id=row["sequence_id"], frame_index=row["frame_index"], variants=variants)
        )
        print(
            row["sequence_id"],
            {k: (v["retained"], v["joint_pass"]) for k, v in variants.items()},
            flush=True,
        )
    summary = {
        v: dict(
            joint_passes=sum(r["variants"][v]["joint_pass"] for r in rows),
            pairs=len(rows),
            reciprocal_pairs=sum(r["variants"][v]["retained"] for r in rows),
        )
        for v in ("baseline", "tiled")
    }
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed source: {p}")
    result = dict(
        summary=summary,
        rows=rows,
        source_reconstruction=replay,
        input_and_source_sha256=hashes,
        radius_pixels=3.0,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Train16 box proxy, not pixel-ground-truth qualification.",
            "Reciprocal false matches can survive this filter.",
            "No reverse control matches were saved; unrelated-pair filter specificity is untested.",
        ],
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
