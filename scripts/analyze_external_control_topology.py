"""Measure orientation of frozen control triangles without filtering references."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_resolution import protocol

SOURCE = Path("experiments/registration_external_resolution_gpu_01/report.json")
SOURCE_SHA = "2e87d1b7ad9c86eea43c75de2e65aa1885e7c3d78674eeafca05b9d73c055a92"


def triangle_geometry(source, target, queries):
    source, target, queries = (np.asarray(x, dtype=float) for x in (source, target, queries))
    if (
        source.ndim != 2
        or source.shape[1] != 2
        or source.shape != target.shape
        or queries.ndim != 2
        or queries.shape[1] != 2
        or not all(np.isfinite(x).all() for x in (source, target, queries))
    ):
        raise ValueError("finite paired 2D controls and queries required")
    triangulation = Delaunay(source)
    a, b = source[triangulation.simplices], target[triangulation.simplices]
    dx = np.stack((a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=-1)
    dy = np.stack((b[:, 1] - b[:, 0], b[:, 2] - b[:, 0]), axis=-1)
    determinant = np.linalg.det(dy) / np.linalg.det(dx)
    cells = triangulation.find_simplex(queries)
    inside = cells >= 0
    nonpositive = np.zeros(len(queries), dtype=bool)
    nonpositive[inside] = determinant[cells[inside]] <= 0
    return dict(
        triangles=triangulation.simplices,
        determinant=determinant,
        query_simplex=cells,
        query_inside=inside,
        query_nonpositive=nonpositive,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config, _, manifest, hashes = protocol()
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("frozen report differs")
    report = json.loads(SOURCE.read_text())
    if report["input_and_source_sha256"] != hashes:
        raise ValueError("provenance differs")
    keys = [(r["pair"], r["long_side"]) for r in report["rows"]]
    expected = {(p, s) for p in manifest["pairs"] for s in (640, 1280)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("case inventory differs")
    hashes.update({str(SOURCE): SOURCE_SHA, str(Path(__file__)): file_sha256(__file__)})
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in report["rows"]:
        path = SOURCE.parent / row["artifact"]
        if path.resolve().parent != SOURCE.parent.resolve() or file_sha256(path) != row["sha256"]:
            raise ValueError("unsafe or changed matches")
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as f:
            ids = row["fit"]["control_match_indices"]
            source, target = f["rgb"][ids], f["thermal"][ids]
            errors = f["errors"].copy()
        queries = np.loadtxt(Path(config["cache"]) / row["pair"] / "points_rgb.txt")
        arrays = triangle_geometry(source, target, queries)
        artifact = args.out_dir / row["artifact"]
        np.savez_compressed(
            artifact,
            **arrays,
            source_controls=source,
            target_controls=target,
            reference_queries=queries,
            tps_errors=errors,
        )
        bad = arrays["query_nonpositive"]
        rows.append(
            dict(
                pair=row["pair"],
                long_side=row["long_side"],
                controls=len(ids),
                triangles=len(arrays["triangles"]),
                nonpositive_triangles=int((arrays["determinant"] <= 0).sum()),
                references=len(queries),
                outside_hull=int((~arrays["query_inside"]).sum()),
                references_in_nonpositive_triangles=int(bad.sum()),
                finite_tps_error_over10=int((np.isfinite(errors) & (errors > 10)).sum()),
                unavailable_tps=int((~np.isfinite(errors)).sum()),
                finite_tps_error_over10_in_nonpositive=int(
                    (np.isfinite(errors) & (errors > 10) & bad).sum()
                ),
                artifact=artifact.name,
                sha256=file_sha256(artifact),
            )
        )
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Linear control-triangle orientation is not a TPS Jacobian test.",
            "Association with error is not unique causal attribution.",
            "No matches or references removed and no model fitted to GT.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
