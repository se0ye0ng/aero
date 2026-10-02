"""Locate loss of interpolation support without treating a hull as correctness."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from aero_ir.utils.manifest import file_sha256
from scripts.analyze_external_control_topology import SOURCE, SOURCE_SHA
from scripts.probe_external_resolution import protocol


def support_masks(raw, controls, queries):
    return dict(
        raw_hull=Delaunay(raw).find_simplex(queries) >= 0,
        control_hull=Delaunay(controls).find_simplex(queries) >= 0,
        nearest_raw_rgb_px=cKDTree(raw).query(queries)[0],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config, _, manifest, hashes = protocol()
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("source changed")
    source = json.loads(SOURCE.read_text())
    if hashes != source["input_and_source_sha256"]:
        raise ValueError("provenance changed")
    cases = [r for r in source["rows"] if r["long_side"] == 640]
    if len(cases) != len(manifest["pairs"]) or {r["pair"] for r in cases} != set(manifest["pairs"]):
        raise ValueError("incomplete or duplicate panel")
    hashes.update(
        {
            str(SOURCE): SOURCE_SHA,
            str(Path(__file__)): file_sha256(__file__),
            "scripts/analyze_external_control_topology.py": file_sha256(
                "scripts/analyze_external_control_topology.py"
            ),
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in cases:
        path = SOURCE.parent / row["artifact"]
        if path.resolve().parent != SOURCE.parent.resolve() or file_sha256(path) != row["sha256"]:
            raise ValueError("unsafe or changed matches")
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as f:
            raw, errors = f["rgb"].copy(), f["errors"].copy()
        ids = row["fit"]["control_match_indices"]
        queries = np.loadtxt(Path(config["cache"]) / row["pair"] / "points_rgb.txt")
        masks = support_masks(raw, raw[ids], queries)
        unavailable = ~np.isfinite(errors)
        artifact = args.out_dir / row["artifact"]
        np.savez_compressed(artifact, **masks, unavailable=unavailable, reference_queries=queries)
        rows.append(
            dict(
                pair=row["pair"],
                landmarks=len(queries),
                unavailable=int(unavailable.sum()),
                unavailable_inside_raw=int((unavailable & masks["raw_hull"]).sum()),
                unavailable_outside_raw=int((unavailable & ~masks["raw_hull"]).sum()),
                unavailable_inside_controls=int((unavailable & masks["control_hull"]).sum()),
                artifact=artifact.name,
                sha256=file_sha256(artifact),
            )
        )
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed input: {path}")
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        limitation=(
            "Raw hull membership does not establish correct local matches or achievable accuracy."
        ),
    )
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
