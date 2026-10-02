"""Exploratory GT-free control pruning; orientation is not correspondence truth."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, QhullError

from aero_ir.utils.manifest import file_sha256
from scripts.analyze_external_control_topology import SOURCE, SOURCE_SHA, triangle_geometry
from scripts.probe_external_resolution import aggregate, protocol, reference_score


def prune_controls(source, target, confidence, minimum=6):
    """Remove highest bad-incidence fraction, then count, then lowest confidence."""
    ids = np.arange(len(source))
    history = []
    while len(ids) >= minimum:
        try:
            g = triangle_geometry(source[ids], target[ids], np.empty((0, 2)))
        except QhullError:
            return ids, history, "degenerate"
        bad = g["determinant"] <= 0
        if not bad.any():
            return ids, history, "positive_control_triangles"
        if len(ids) == minimum:
            break
        triangles = g["triangles"]
        degree = np.bincount(triangles.ravel(), minlength=len(ids))
        bad_degree = np.bincount(triangles[bad].ravel(), minlength=len(ids))
        fraction = bad_degree / np.maximum(degree, 1)
        candidates = np.flatnonzero(bad_degree)
        index = min(
            candidates, key=lambda i: (-fraction[i], -bad_degree[i], confidence[ids[i]], ids[i])
        )
        history.append(dict(removed_control=int(ids[index]), bad_triangles=int(bad.sum())))
        ids = np.delete(ids, index)
    return ids, history, "unresolved_at_minimum"


def selected_tps(source, target, source_size, target_size, smoothing):
    x = (source + 0.5) / max(source_size)
    fit = RBFInterpolator(
        x,
        (target + 0.5) / max(target_size),
        kernel="thin_plate_spline",
        degree=1,
        smoothing=smoothing,
    )
    hull = Delaunay(x)

    def predict(query):
        q = (np.asarray(query) + 0.5) / max(source_size)
        result = fit(q) * max(target_size) - 0.5
        valid = (hull.find_simplex(q) >= 0) & np.isfinite(result).all(1)
        valid &= (result >= 0).all(1) & (result <= np.asarray(target_size) - 1).all(1)
        result[~valid] = np.nan
        return result

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config, tps, manifest, hashes = protocol()
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("baseline changed")
    baseline = json.loads(SOURCE.read_text())
    if hashes != baseline["input_and_source_sha256"]:
        raise ValueError("baseline provenance changed")
    keys = [(r["pair"], r["long_side"]) for r in baseline["rows"]]
    expected = {(p, s) for p in manifest["pairs"] for s in (640, 1280)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("case inventory changed")
    for path in (SOURCE, Path(__file__), Path("scripts/analyze_external_control_topology.py")):
        hashes[str(path)] = file_sha256(path)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in baseline["rows"]:
        path = SOURCE.parent / row["artifact"]
        if path.resolve().parent != SOURCE.parent.resolve() or file_sha256(path) != row["sha256"]:
            raise ValueError("unsafe/changed match artifact")
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as f:
            controls = np.asarray(row["fit"]["control_match_indices"])
            x, y, c = f["rgb"][controls], f["thermal"][controls], f["confidence"][controls]
            original_errors = f["errors"].copy()
        keep, history, status = prune_controls(x, y, c, tps["minimum_controls"])
        root, pair = Path(config["cache"]), row["pair"]
        with Image.open(root / pair / "V.JPG") as im:
            rgb_size = im.size
        with Image.open(root / pair / "T.JPG") as im:
            thr_size = im.size
        warp = None
        if status == "positive_control_triangles":
            try:
                warp = selected_tps(x[keep], y[keep], rgb_size, thr_size, tps["tps_smoothing"])
            except (QhullError, ValueError, np.linalg.LinAlgError):
                status = "fit_failed"
        errors, summary = reference_score(
            root, pair, rgb_size[::-1], thr_size[::-1], warp, tps["thresholds_thermal_file_pixels"]
        )
        artifact = args.out_dir / row["artifact"]
        np.savez_compressed(
            artifact,
            errors=errors,
            original_errors=original_errors,
            retained_match_indices=controls[keep],
        )
        rows.append(
            dict(
                pair=pair,
                long_side=row["long_side"],
                summary=summary,
                original_summary=row["summary"],
                controls_before=len(x),
                controls_after=len(keep),
                status=status,
                removal_history=history,
                artifact=artifact.name,
                sha256=file_sha256(artifact),
            )
        )
        print(pair, row["long_side"], len(x), len(keep), status, summary, flush=True)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed dependency: {path}")
    result = dict(
        rows=rows,
        aggregate=aggregate(rows),
        baseline_aggregate=baseline["aggregate"],
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Exploratory image-only heuristic on observed pairs.",
            "Positive control triangles do not guarantee positive TPS Jacobians.",
            "All unsupported references remain failures; no GT used for pruning.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
