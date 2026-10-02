"""Preserve frozen grid controls and add hull vertices of locally accepted matches."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

from aero_ir.utils.manifest import file_sha256
from scripts.analyze_external_control_topology import SOURCE, SOURCE_SHA
from scripts.probe_external_orientation_repair import selected_tps
from scripts.probe_external_resolution import aggregate, protocol, reference_score, validate_replay


def accepted_matches(rgb, thermal, confidence, rgb_size, thermal_size, config):
    """Replay the frozen leave-one-out local affine consensus, before grid reduction."""
    rgb, thermal, confidence = [np.asarray(a, dtype=np.float64) for a in (rgb, thermal, confidence)]
    order = np.argsort(-confidence, kind="stable")
    _, unique = np.unique(rgb[order], axis=0, return_index=True)
    chosen = order[np.sort(unique)]
    x, y = (rgb[chosen] + 0.5) / max(rgb_size), (thermal[chosen] + 0.5) / max(thermal_size)
    k = min(config["neighbors"], len(x) - 1)
    neighbors = cKDTree(x).query(x, k=k + 1)[1]
    residual = np.full(len(x), np.inf)
    for i in range(len(x)):
        ids = neighbors[i][neighbors[i] != i][:k]
        design = np.column_stack((x[ids] - x[i], np.ones(len(ids))))
        target, selected = y[ids], np.arange(len(ids))
        for _ in range(config["trim_iterations"]):
            coef, _, rank, _ = np.linalg.lstsq(design[selected], target[selected], rcond=None)
            if rank < 3:
                break
            errors = np.linalg.norm(design @ coef - target, axis=1)
            count = max(4, int(np.ceil(config["trim_fraction"] * len(ids))))
            selected = np.argsort(errors, kind="stable")[:count]
        else:
            coef, _, rank, _ = np.linalg.lstsq(design[selected], target[selected], rcond=None)
            if rank == 3:
                residual[i] = np.linalg.norm(coef[-1] - y[i]) * max(thermal_size)
    threshold = max(
        config["agreement_min_thermal_pixels"],
        config["agreement_thermal_long_side_fraction"] * max(thermal_size),
    )
    accepted = chosen[residual <= threshold]
    occupied, grid = set(), []
    for idx in accepted:
        cell = tuple(
            np.floor((rgb[idx] + 0.5) / np.asarray(rgb_size) * config["control_grid"]).astype(int)
        )
        if cell not in occupied:
            grid.append(int(idx))
            occupied.add(cell)
    return accepted, np.asarray(grid, dtype=int)


def augment_boundary(rgb, accepted, original):
    boundary = accepted[ConvexHull(rgb[accepted]).vertices]
    additions = [int(i) for i in boundary if i not in set(original)]
    return np.concatenate((original, np.asarray(additions, dtype=int)))


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
    if baseline["input_and_source_sha256"] != hashes:
        raise ValueError("provenance changed")
    cases = [r for r in baseline["rows"] if r["long_side"] == 640]
    if len(cases) != len(manifest["pairs"]) or {r["pair"] for r in cases} != set(manifest["pairs"]):
        raise ValueError("case inventory differs")
    for path in (
        SOURCE,
        Path(__file__),
        Path("scripts/probe_external_orientation_repair.py"),
        Path("scripts/analyze_external_control_topology.py"),
    ):
        hashes[str(path)] = file_sha256(path)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in cases:
        path = SOURCE.parent / row["artifact"]
        if path.resolve().parent != SOURCE.parent.resolve() or file_sha256(path) != row["sha256"]:
            raise ValueError("unsafe or changed matches")
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as f:
            a = dict(f)
        root, pair = Path(config["cache"]), row["pair"]
        with Image.open(root / pair / "V.JPG") as im:
            rgb_size = im.size
        with Image.open(root / pair / "T.JPG") as im:
            thr_size = im.size
        accepted, grid = accepted_matches(
            a["rgb"], a["thermal"], a["confidence"], rgb_size, thr_size, tps
        )
        np.testing.assert_array_equal(grid, row["fit"]["control_match_indices"])
        if len(accepted) != row["fit"]["accepted_matches"]:
            raise ValueError("consensus replay differs")
        original_warp = selected_tps(
            a["rgb"][grid], a["thermal"][grid], rgb_size, thr_size, tps["tps_smoothing"]
        )
        e, s = reference_score(
            root,
            pair,
            rgb_size[::-1],
            thr_size[::-1],
            original_warp,
            tps["thresholds_thermal_file_pixels"],
        )
        validate_replay({}, s, e, {}, row["summary"], a["errors"], pair)
        ids = augment_boundary(a["rgb"], accepted, grid)
        warp = selected_tps(
            a["rgb"][ids], a["thermal"][ids], rgb_size, thr_size, tps["tps_smoothing"]
        )
        errors, summary = reference_score(
            root, pair, rgb_size[::-1], thr_size[::-1], warp, tps["thresholds_thermal_file_pixels"]
        )
        artifact = args.out_dir / row["artifact"]
        np.savez_compressed(
            artifact,
            errors=errors,
            original_errors=a["errors"],
            accepted_indices=accepted,
            original_controls=grid,
            expanded_controls=ids,
        )
        rows.append(
            dict(
                pair=pair,
                long_side=640,
                summary=summary,
                original_summary=row["summary"],
                controls_before=len(grid),
                controls_after=len(ids),
                artifact=artifact.name,
                sha256=file_sha256(artifact),
            )
        )
        print(pair, len(grid), len(ids), summary, flush=True)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    report = dict(
        rows=rows,
        aggregate=aggregate(rows),
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Exploratory same-panel boundary policy, not qualification.",
            "Original controls retained; locally accepted matches may still be wrong.",
            "All references retained. No GT used in control selection.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
