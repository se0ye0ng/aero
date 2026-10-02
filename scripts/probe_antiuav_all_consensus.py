"""All-consensus piecewise-affine interpolation; no GT-based control selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_external_boundary_controls import accepted_matches
from scripts.verify_antiuav_tiled_matching import safe_artifact, verify


def local_interpolator(p, q, target_size):
    """Continuous piecewise affine map, no hull extrapolation or out-of-frame output.

    Does not claim positive Jacobians, inverse consistency or physical correctness.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape != q.shape:
        raise ValueError("paired Nx2 arrays required")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("finite arrays required")
    if len(p) < 6:
        return None
    try:
        interp = LinearNDInterpolator(p, q)
    except QhullError:
        return None

    def predict(points):
        result = np.asarray(interp(points))
        valid = np.isfinite(result).all(1)
        valid &= (result >= 0).all(1) & (result <= np.asarray(target_size) - 1).all(1)
        result[~valid] = np.nan
        return result

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    source_check = verify(args.report)
    source = json.loads(args.report.read_text())
    settings_path = Path("configs/experiment/registration_external_local_warp_cpu.yaml")
    settings = yaml.safe_load(settings_path.read_text())
    files = [
        args.report,
        settings_path,
        Path(__file__),
        Path("scripts/probe_external_boundary_controls.py"),
        Path("scripts/verify_antiuav_tiled_matching.py"),
        Path("scripts/probe_antiuav_local_warp.py"),
    ]
    hashes = {str(p): file_sha256(p) for p in files}
    rows = []
    for row in source["rows"]:
        path = safe_artifact(args.report.parent, row)
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as a:
            for variant, prefix in (("baseline", "baseline_"), ("tiled", "")):
                directions = []
                for d, names in enumerate(
                    (
                        ("points0", "points1", "confidence"),
                        ("reverse0", "reverse1", "reverse_confidence"),
                    )
                ):
                    p, q, c = [a[prefix + k] for k in names]
                    sizes = row["sizes"] if d == 0 else row["sizes"][::-1]
                    old = row["scores"][variant]["directions"][d]["fit"]
                    accepted = np.empty(0, dtype=int)
                    if old["unique_rgb_matches"] >= settings["minimum_controls"]:
                        accepted, grid = accepted_matches(p, q, c, *sizes, settings)
                        np.testing.assert_array_equal(grid, old["control_match_indices"])
                        if len(accepted) != old["accepted_matches"]:
                            raise ValueError("consensus reconstruction differs")
                    warp = local_interpolator(p[accepted], q[accepted], sizes[1])
                    boxes = row["boxes_xyxy"] if d == 0 else row["boxes_xyxy"][::-1]
                    directions.append(
                        dict(
                            accepted_indices=accepted.tolist(),
                            fit_available=warp is not None,
                            iou=corner_iou(warp, *boxes),
                        )
                    )
                passed = all(x["iou"] is not None and x["iou"] >= 0.6 for x in directions)
                rows.append(
                    dict(
                        sequence_id=row["sequence_id"],
                        variant=variant,
                        directions=directions,
                        joint_pass=passed,
                    )
                )
        print(row["sequence_id"], [(r["variant"], r["joint_pass"]) for r in rows[-2:]], flush=True)
    summary = {
        v: dict(pairs=16, joint_passes=sum(r["joint_pass"] for r in rows if r["variant"] == v))
        for v in ("baseline", "tiled")
    }
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed source: {path}")
    result = dict(
        rows=rows,
        summary=summary,
        input_and_source_sha256=hashes,
        source_verification=source_check,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Train16 box proxy only; no independent pixel accuracy.",
            "Consensus acceptance unchanged; all accepted points used.",
            "No inverse or topology guarantee; unsupported corners fail.",
        ],
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
