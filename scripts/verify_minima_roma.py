"""Reconstruct RoMa scores from saved dense fields; no neural inference replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.probe_minima_roma_gpu import evaluate, protocol
from scripts.verify_antiuav_tiled_matching import canonical_hashes, check_score, safe_artifact


def verify(path):
    report = json.loads(path.read_text())
    if report["registration_qualified"] or report["generator_training_approved"]:
        raise ValueError("diagnostic cannot authorize qualification")
    required, cases, metadata = protocol()
    actual = canonical_hashes(report["input_and_source_sha256"])
    if actual != canonical_hashes(required):
        raise ValueError("provenance inventory differs")
    for name, digest in actual.items():
        if file_sha256(name) != digest:
            raise ValueError(f"source changed: {name}")
    cases = {(r["sequence_id"], r["frame_index"]): r for r in cases}
    keys = [(r["sequence_id"], r["frame_index"]) for r in report["rows"]]
    if report["pairs"] != 16 or len(keys) != 16 or len(set(keys)) != 16 or set(keys) != set(cases):
        raise ValueError("missing or duplicate pair")
    count = 0
    shapes_by_key = {}
    for row in report["rows"]:
        key = row["sequence_id"], row["frame_index"]
        case = cases[key]
        if row["header_rows"] != case["header_rows"]:
            raise ValueError("header offsets changed")
        np.testing.assert_allclose(row["boxes_xyxy"], case["boxes_xyxy"], atol=1e-12, rtol=0)
        shapes = [
            [
                metadata[key[0]][m]["resized_shape"][0] - case["header_rows"][d],
                metadata[key[0]][m]["resized_shape"][1],
            ]
            for d, m in enumerate(("visible", "infrared"))
        ]
        if row["crop_shapes"] != shapes:
            raise ValueError("crop dimensions changed")
        shapes_by_key[key] = shapes
        # Image values are unused in score reconstruction; only their dimensions matter.
        images = [np.empty((*shape, 3), dtype=np.uint8) for shape in shapes]
        with np.load(safe_artifact(path.parent, row), allow_pickle=False) as a:
            score = evaluate(
                a["warp"], a["certainty"], images, case["header_rows"], case["boxes_xyxy"]
            )
        old = row["scores"]
        if score["joint_pass"] != old["joint_pass"]:
            raise ValueError("pass decision differs")
        for field in ("iou", "certainty_ge_half_fraction"):
            if len(old[field]) != 2:
                raise ValueError("missing directional score")
            for x, y in zip(score[field], old[field], strict=True):
                check_score(x, y)
        count += int(score["joint_pass"])
    if report["joint_passes"] != count:
        raise ValueError("summary differs")
    if sorted(c["kind"] for c in report["controls"]) != ["known_shift", "unrelated"]:
        raise ValueError("missing or duplicate control")
    # Generation uses frozen panel order, not report order.
    first_key = next(iter(cases))
    h, w = shapes_by_key[first_key][0]
    for row in report["controls"]:
        with np.load(safe_artifact(path.parent, row), allow_pickle=False) as a:
            maps = split_dense(a["warp"], a["certainty"])
            if len(row["certainty_ge_half_fraction"]) != 2:
                raise ValueError("missing control direction")
            for (_, c), expected in zip(maps, row["certainty_ge_half_fraction"], strict=True):
                check_score(float((c >= 0.5).mean()), expected)
            if row["kind"] == "known_shift":
                yy, xx = np.mgrid[16 : h - 16 : 16, 16 : w - 16 : 16]
                query = np.c_[xx.ravel(), yy.ravel()]
                pred = dense_predictor(*maps[0], (w, h), (w, h))(query)
                error = np.linalg.norm(pred - query - [8, -8], axis=1)
                if row["queries"] != len(query) or row["supported"] != int(
                    np.isfinite(error).sum()
                ):
                    raise ValueError("control support differs")
                check_score(float(np.mean(error <= 3)), row["pck3_all"])
    return dict(
        ok=True,
        pairs=16,
        controls=2,
        joint_passes=count,
        report_sha256=file_sha256(path),
        neural_inference_replayed=False,
        registration_qualified=False,
        generator_training_approved=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.report), indent=2))


if __name__ == "__main__":
    main()
