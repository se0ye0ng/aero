"""Post-hoc bounded residual acceptance; preserves coarse map when rejected."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.prewarp import compose, field_predictor
from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_roma_prewarp_motion import REPORT, SHA
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_roma_cycle_rejection import admission
from scripts.probe_roma_prewarp_gpu import INPUT, protocol, scores
from scripts.verify_antiuav_tiled_matching import safe_artifact


def guarded(forward, reverse, size, max_pixels):
    if not np.isfinite(max_pixels) or max_pixels <= 0:
        raise ValueError("positive finite movement bound required")

    def predict(query):
        proposed, accepted, _ = admission(forward, reverse, query, size, size)
        accepted &= np.linalg.norm(proposed - query, axis=1) <= max_pixels
        output = np.array(query, dtype=float, copy=True)
        output[accepted] = proposed[accepted]
        return output

    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if file_sha256(REPORT) != SHA:
        raise ValueError("inference report changed")
    manifest, hashes = protocol()
    report = json.loads(REPORT.read_text())
    if report["input_and_source_sha256"] != hashes:
        raise ValueError("source inventory changed")
    if [r["sequence_id"] for r in report["rows"]] != [r["sequence_id"] for r in manifest["rows"]]:
        raise ValueError("pair inventory differs")
    rows = []
    for row, original in zip(report["rows"], manifest["rows"], strict=True):
        path = safe_artifact(REPORT.parent, row)
        hashes[str(path)] = row["sha256"]
        with (
            np.load(path, allow_pickle=False) as saved,
            np.load(safe_artifact(INPUT.parent, original), allow_pickle=False) as data,
        ):
            if scores(saved["warp"], saved["certainty"], original, data) != row["scores"]:
                raise ValueError("baseline score reconstruction differs")
            sizes, tops, boxes = (
                original["image_sizes"],
                original["header_rows"],
                original["boxes_xyxy"],
            )
            coarse = [
                field_predictor(data[name], sizes[d], sizes[1 - d])
                for d, name in enumerate(("rgb_to_ir_field", "ir_to_rgb_field"))
            ]
            crop = (sizes[1][0], sizes[1][1] - tops[1])
            residual = [
                dense_predictor(f, np.ones_like(c), crop, crop, tops[1], tops[1])
                for f, c in split_dense(saved["warp"], saved["certainty"])
            ]
            conditions = {}
            for bound in (0, 1, 3, 5):
                refinement = (
                    [lambda q: q, lambda q: q]
                    if bound == 0
                    else [guarded(residual[d], residual[1 - d], sizes[1], bound) for d in range(2)]
                )
                predictors = compose(
                    *coarse, *refinement, data["observed"], rgb_top=tops[0], ir_top=tops[1]
                )
                ious = [corner_iou(fn, boxes[d], boxes[1 - d]) for d, fn in enumerate(predictors)]
                conditions[str(bound)] = dict(
                    iou=ious, both_ge_06=all(v is not None and v >= 0.6 for v in ious)
                )
            rows.append(dict(sequence_id=row["sequence_id"], conditions=conditions))
    counts = {
        str(b): sum(r["conditions"][str(b)]["both_ge_06"] for r in rows) for b in (0, 1, 3, 5)
    }
    for path in (
        REPORT,
        Path(__file__),
        Path("scripts/analyze_roma_prewarp_motion.py"),
        Path("scripts/probe_roma_cycle_rejection.py"),
    ):
        hashes[str(path)] = file_sha256(path)
    result = dict(
        rows=rows,
        both_box_pass_counts=counts,
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        protocol="0=coarse; bounds1,3,5 IR pixels; cycle<=1; positive bidirectional Jacobians",
        limitations=[
            "Post-hoc sensitivity analysis; no threshold chosen by GT scores.",
            "Pointwise switching can introduce discontinuities and break inverses.",
            "Preserving initial predictions does not repair initial errors.",
            "Box proxy is not independent pixel GT or export authorization.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
