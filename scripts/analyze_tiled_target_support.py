"""Post-hoc box membership audit, never a box-guided correspondence selector."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_tiled_reciprocity import reciprocal_ids
from scripts.verify_antiuav_tiled_matching import safe_artifact, verify


def in_box(points, box):
    points, box = np.asarray(points), np.asarray(box)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("positive finite xyxy box required")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("finite Nx2 points required")
    return ((points >= box[:2]) & (points < box[2:])).all(axis=1)


def membership(p, q, boxes):
    if len(p) != len(q):
        raise ValueError("paired endpoints required")
    src, dst = in_box(p, boxes[0]), in_box(q, boxes[1])
    return dict(
        total=len(p),
        source_box=int(src.sum()),
        target_box=int(dst.sum()),
        both_boxes=int((src & dst).sum()),
        source_box_to_outside=int((src & ~dst).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    verification = verify(args.report)
    report = json.loads(args.report.read_text())
    files = [
        args.report,
        Path(__file__),
        Path("scripts/verify_antiuav_tiled_matching.py"),
        Path("scripts/probe_antiuav_tiled_reciprocity.py"),
    ]
    hashes = {str(p): file_sha256(p) for p in files}
    rows = []
    for row in report["rows"]:
        path = safe_artifact(args.report.parent, row)
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as a:
            for name, prefix in (("baseline", "baseline_"), ("tiled", "")):
                p, q, rp, rq = [
                    a[prefix + k] for k in ("points0", "points1", "reverse0", "reverse1")
                ]
                fi, ri = reciprocal_ids(p, q, rp, rq)
                for direction, (x, y, keep, boxes) in enumerate(
                    (
                        (p, q, fi, row["boxes_xyxy"]),
                        (rp, rq, ri, row["boxes_xyxy"][::-1]),
                    )
                ):
                    fit = row["scores"][name]["directions"][direction]["fit"]
                    controls = np.asarray(fit.get("control_match_indices", []), dtype=int)
                    rows.append(
                        dict(
                            sequence_id=row["sequence_id"],
                            variant=name,
                            direction=direction,
                            raw=membership(x, y, boxes),
                            reciprocal=membership(x[keep], y[keep], boxes),
                            controls=membership(x[controls], y[controls], boxes),
                            fit_status=fit["status"],
                        )
                    )
    summary = {}
    for name in ("baseline", "tiled"):
        selected = [r for r in rows if r["variant"] == name]
        summary[name] = {}
        for stage in ("raw", "reciprocal", "controls"):
            summary[name][stage] = dict(
                directions=len(selected),
                directions_with_source=sum(r[stage]["source_box"] > 0 for r in selected),
                directions_with_both=sum(r[stage]["both_boxes"] > 0 for r in selected),
                source_points=sum(r[stage]["source_box"] for r in selected),
                both_box_points=sum(r[stage]["both_boxes"] for r in selected),
                source_to_outside=sum(r[stage]["source_box_to_outside"] for r in selected),
            )
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed input: {p}")
    result = dict(
        summary=summary,
        rows=rows,
        input_and_source_sha256=hashes,
        source_verification=verification,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Post-hoc train16 box audit; boxes never change matches or transforms.",
            "Both endpoints in boxes is not correct semantic/pixel correspondence.",
            "Boxes include background; absent in-box controls alone does not prove "
            "interpolation impossible.",
            "Counts include correlated matches across tiles; not independent samples.",
        ],
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
