"""Bound box-corner qualification coverage before spending GPU compute."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.utils.manifest import file_sha256


def covers_box_corners(crop, box, header):
    if crop is None:
        return False
    b = np.asarray(box, dtype=float) - [0, header, 0, header]
    c = np.asarray(crop, dtype=float)
    # Match the frozen scorer's corner locations and inclusive pixel-centre support.
    return bool((b[:2] >= c[:2]).all() and (b[2:] <= c[2:] - 1).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(args.plan) != "b9b9b47d9e2d4d2342e040b7436c7e122c505cdb7e092d58fb4940bc011ac86c":
        raise ValueError("frozen crop plan changed")
    plan = json.loads(args.plan.read_text())
    for path, digest in plan["input_and_source_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    rows = [
        dict(
            sequence_id=row["sequence_id"],
            source_corner_support=[
                covers_box_corners(
                    row["crop_bounds"][d], row["boxes_xyxy"][d], row["header_rows"][d]
                )
                for d in range(2)
            ],
        )
        for row in plan["rows"]
    ]
    result = dict(
        rows=rows,
        pair_count=len(rows),
        maximum_possible_joint_box_passes_without_extrapolation=sum(
            all(r["source_corner_support"]) for r in rows
        ),
        identity_box_proxy_passes=plan["identity_box_proxy_both_ge_06"],
        source_sha256=file_sha256(__file__),
        plan_sha256=file_sha256(args.plan),
        gpu_inference_executed=False,
        registration_qualified=False,
        generator_training_approved=False,
        conclusion="Single-component motion crop is not a complete repair; do not promote.",
        limitation=(
            "This is a coverage ceiling for this crop-only model, "
            "not a matching accuracy estimate."
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
