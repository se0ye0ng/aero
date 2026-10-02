"""CPU reconstruction of saved tiled-match geometry, not neural inference replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_external_local_warp import fit_image_warp


def validate_tile_inventory(flags):
    rows = flags["records"]
    keys = [(x["tile0"], x["tile1"]) for x in rows]
    if (
        flags["tile_pairs"] != 81
        or len(keys) != 81
        or set(keys) != {(i, j) for i in range(9) for j in range(9)}
    ):
        raise ValueError("missing or duplicate tile pair")
    if any(type(x["retained"]) is not int or x["retained"] < 0 for x in rows):
        raise ValueError("invalid retained match count")
    if sum(x["retained"] for x in rows) != flags["before_exact_dedup"]:
        raise ValueError("pre-dedup count differs")


def safe_artifact(root, row):
    path = root / row["artifact"]
    if path.resolve().parent != root.resolve() or file_sha256(path) != row["sha256"]:
        raise ValueError("unsafe or changed artifact")
    return path


def check_score(actual, expected):
    if actual is None or expected is None:
        if actual != expected:
            raise ValueError("supported/unsupported score differs")
    elif not np.isfinite([actual, expected]).all() or abs(actual - expected) > 1e-8:
        raise ValueError("recomputed IoU differs")


def canonical_hashes(hashes):
    """Compare identities by resolved path; reject conflicting alias digests."""
    result = {}
    for name, digest in hashes.items():
        path = Path(name).resolve()
        if path in result and result[path] != digest:
            raise ValueError(f"conflicting provenance aliases: {name}")
        result[path] = digest
    return result


def verify(report_path):
    r = json.loads(report_path.read_text())
    if r["registration_qualified"] or r["generator_training_approved"]:
        raise ValueError("diagnostic cannot authorize qualification")
    hashes = r["input_and_source_sha256"]
    identities = canonical_hashes(hashes)
    config_path = Path("configs/experiment/registration_rgb_resolution_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    base = Path(config["baseline"])
    if file_sha256(base) != config["baseline_sha256"]:
        raise ValueError("baseline identity changed")
    required = json.loads(base.read_text())["input_and_source_sha256"]
    if any(identities.get(p) != digest for p, digest in canonical_hashes(required).items()):
        raise ValueError("missing baseline provenance")
    for p in (
        str(config_path),
        str(base),
        "scripts/probe_antiuav_tiled_matching.py",
        "scripts/run_antiuav_tiled_matching_gpu.sh",
        "src/aero_ir/registration/tiled_matching.py",
    ):
        if Path(p).resolve() not in identities:
            raise ValueError(f"missing required provenance: {p}")
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed input/source: {p}")
    source = json.loads(Path(config["source_report"]).read_text())
    cases = {
        (x["sequence_id"], x["frame_index"]): x
        for x in source["rows"]
        if x["condition"] == "input_header_crop"
    }
    keys = [(x["sequence_id"], x["frame_index"]) for x in r["rows"]]
    if len(keys) != 16 or len(set(keys)) != 16 or set(keys) != set(cases):
        raise ValueError("missing or duplicate evaluation pair")
    metadata = {x["sequence_id"]: x["inputs"] for x in source["inputs"]}
    settings = yaml.safe_load(Path(config["local_config"]).read_text())
    totals = dict(baseline=0, tiled=0)
    for row in r["rows"]:
        case = cases[(row["sequence_id"], row["frame_index"])]
        np.testing.assert_allclose(row["boxes_xyxy"], case["boxes_xyxy"], atol=1e-4, rtol=0)
        sizes = [
            list(metadata[row["sequence_id"]][name]["resized_shape"][::-1])
            for name in ("visible", "infrared")
        ]
        if row["sizes"] != sizes:
            raise ValueError("working-grid dimensions differ")
        if len(row["flags"]) != 2:
            raise ValueError("missing directional flags")
        for flags in row["flags"]:
            validate_tile_inventory(flags)
        with np.load(safe_artifact(report_path.parent, row), allow_pickle=False) as data:
            for variant, prefix in (("baseline", "baseline_"), ("tiled", "")):
                values = []
                if len(row["scores"][variant]["directions"]) != 2:
                    raise ValueError("missing directional score")
                for d, names in enumerate(
                    (
                        ("points0", "points1", "confidence"),
                        ("reverse0", "reverse1", "reverse_confidence"),
                    )
                ):
                    arrays = [data[prefix + n] for n in names]
                    warp, info = fit_image_warp(
                        *arrays, *(sizes if d == 0 else sizes[::-1]), settings
                    )
                    old = row["scores"][variant]["directions"][d]
                    if info != old["fit"] or len(arrays[0]) != old["matches"]:
                        raise ValueError("fit or match count differs")
                    if (
                        variant == "tiled"
                        and len(arrays[0]) > row["flags"][d]["before_exact_dedup"]
                    ):
                        raise ValueError("dedup increased count")
                    boxes = row["boxes_xyxy"] if d == 0 else row["boxes_xyxy"][::-1]
                    iou = corner_iou(warp, *boxes)
                    check_score(iou, old["iou"])
                    values.append(iou)
                passed = all(v is not None and v >= config["box_iou_threshold"] for v in values)
                if passed != row["scores"][variant]["joint_pass"]:
                    raise ValueError("pass decision changed")
                totals[variant] += int(passed)
    if totals != r["summary"]:
        raise ValueError("summary differs")
    if sorted(x["kind"] for x in r["controls"]) != ["same_modality_known_shift", "unrelated_pair"]:
        raise ValueError("missing/duplicate controls")
    for row in r["controls"]:
        validate_tile_inventory(row["flags"])
        with np.load(safe_artifact(report_path.parent, row), allow_pickle=False) as a:
            if len(a["points0"]) != row["matches"]:
                raise ValueError("control match count differs")
            if row["kind"] == "same_modality_known_shift":
                errors = np.linalg.norm(a["points1"] - a["points0"] - [8, -8], axis=1)
                actual = float(np.mean(errors <= 3)) if len(errors) else None
                check_score(actual, row["pck3"])
    return dict(
        ok=True,
        verified_pairs=16,
        report_sha256=file_sha256(report_path),
        summary=totals,
        registration_qualified=False,
        neural_inference_replayed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.report), indent=2))


if __name__ == "__main__":
    main()
