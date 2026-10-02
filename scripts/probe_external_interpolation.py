"""Same frozen controls, different interpolation; exploratory, not qualification."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator

from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_resolution import (
    aggregate,
    fit_image_warp,
    protocol,
    reference_score,
    validate_replay,
)

SOURCE = Path("experiments/registration_external_resolution_gpu_01/report.json")
SOURCE_SHA = "2e87d1b7ad9c86eea43c75de2e65aa1885e7c3d78674eeafca05b9d73c055a92"


def linear_warp(rgb, thermal, rgb_size, thermal_size):
    # Same isotropic source normalization as the frozen TPS. No extrapolation.
    scale = float(max(rgb_size))
    fit = LinearNDInterpolator((np.asarray(rgb) + 0.5) / scale, thermal)

    def predict(query):
        result = np.asarray(fit((np.asarray(query) + 0.5) / scale))
        supported = np.isfinite(result).all(1) & (result >= 0).all(1)
        supported &= (result <= np.asarray(thermal_size) - 1).all(1)
        result[~supported] = np.nan
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
        raise ValueError("GPU report changed")
    prior = json.loads(SOURCE.read_text())
    if prior["input_and_source_sha256"] != hashes:
        raise ValueError("GPU input provenance differs")
    expected = {(p, s) for p in manifest["pairs"] for s in (640, 1280)}
    keys = [(r["pair"], r["long_side"]) for r in prior["rows"]]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("case inventory differs")
    hashes.update({str(SOURCE): SOURCE_SHA, str(Path(__file__)): file_sha256(__file__)})
    for row in prior["rows"]:
        path = SOURCE.parent / row["artifact"]
        if path.resolve().parent != SOURCE.parent.resolve() or file_sha256(path) != row["sha256"]:
            raise ValueError("unsafe or changed match artifact")
        hashes[str(path)] = row["sha256"]
    root = Path(config["cache"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in prior["rows"]:
        with np.load(SOURCE.parent / row["artifact"], allow_pickle=False) as f:
            a = dict(f)
        pair = row["pair"]
        with Image.open(root / pair / "V.JPG") as im:
            rgb_size = im.size
        with Image.open(root / pair / "T.JPG") as im:
            thr_size = im.size
        warp, info = fit_image_warp(
            a["rgb"], a["thermal"], a["confidence"], rgb_size, thr_size, tps
        )
        errors, summary = reference_score(
            root, pair, rgb_size[::-1], thr_size[::-1], warp, tps["thresholds_thermal_file_pixels"]
        )
        validate_replay(info, summary, errors, row["fit"], row["summary"], a["errors"], pair)
        ids = np.asarray(info.get("control_match_indices", []), dtype=int)
        alternate = (
            None
            if warp is None
            else linear_warp(a["rgb"][ids], a["thermal"][ids], rgb_size, thr_size)
        )
        linear_errors, linear_summary = reference_score(
            root,
            pair,
            rgb_size[::-1],
            thr_size[::-1],
            alternate,
            tps["thresholds_thermal_file_pixels"],
        )
        common = np.isfinite(errors) & np.isfinite(linear_errors)
        artifact = args.out_dir / row["artifact"]
        np.savez_compressed(
            artifact,
            tps_errors=errors,
            linear_errors=linear_errors,
            control_match_indices=ids,
            common=common,
        )
        result = dict(
            pair=pair,
            long_side=row["long_side"],
            summary=linear_summary,
            tps_summary=summary,
            controls=len(ids),
            artifact=artifact.name,
            sha256=file_sha256(artifact),
            common_landmarks=int(common.sum()),
            common_tps_median=float(np.median(errors[common])) if common.any() else None,
            common_linear_median=float(np.median(linear_errors[common])) if common.any() else None,
        )
        rows.append(result)
        print(pair, row["long_side"], linear_summary, flush=True)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    report = dict(
        rows=rows,
        linear_aggregate=aggregate(rows),
        tps_aggregate=prior["aggregate"],
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Same observed four pairs, not independent confirmation.",
            "Unchanged image-selected controls; no GT-guided fitting.",
            "Failure of both interpolators cannot uniquely identify bad matches.",
            "Sparse support, scene geometry and interpolation can all contribute.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
