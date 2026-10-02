"""Geometry audit of frozen image-only local TPS; does not qualify dense warps."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from aero_ir.registration.warp_geometry_probe import sample_geometry
from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_local_warp import fit_image_warp

SOURCE = Path("experiments/registration_external_local_warp_cpu_01/report.json")
SHA = "0c1d8e35cebdc67f507cf94edc3f2dd7a77782bd2cb149dda36f6394f15d86d8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(SOURCE) != SHA:
        raise ValueError("frozen baseline changed")
    source = json.loads(SOURCE.read_text())
    hashes = source["input_and_source_sha256"].copy()
    for p in (SOURCE, Path(__file__), Path("src/aero_ir/registration/warp_geometry_probe.py")):
        hashes[str(p)] = file_sha256(p)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed dependency: {p}")
    config = yaml.safe_load(
        Path("configs/experiment/registration_external_local_warp_cpu.yaml").read_text()
    )
    base = yaml.safe_load(Path(config["geometry_config"]).read_text())
    root, matches_root = Path(base["cache"]), Path(base["image_only_report"]).parent
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = dict(
        input_and_source_sha256=hashes,
        source_grid_step=16,
        derivative_delta_px=1.0,
        reference_guided_fitting=False,
        reverse_fit="same policy with swapped modalities",
        warning="sampled local geometry, not global injectivity or pixel accuracy",
        registration_qualified=False,
        generator_training_approved=False,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(plan, indent=2, allow_nan=False))
    rows = []
    for row in source["rows"]:
        pair, method = row["pair"], row["method"]
        with Image.open(root / pair / "V.JPG") as im:
            rgb_size = im.size
        with Image.open(root / pair / "T.JPG") as im:
            thr_size = im.size
        path = matches_root / f"{method}_{pair.replace('/', '_')}_matches.npz"
        with np.load(path, allow_pickle=False) as f:
            a = dict(f)
        forward, info = fit_image_warp(
            a["rgb"], a["thermal"], a["confidence"], rgb_size, thr_size, config
        )
        if info != row["fit"]:
            raise ValueError("forward controls differ from original fit")
        reverse, reverse_info = fit_image_warp(
            a["thermal"], a["rgb"], a["confidence"], thr_size, rgb_size, config
        )
        arrays, stats = sample_geometry(forward, reverse, rgb_size)
        artifact = args.out_dir / f"{method}_{pair.replace('/', '_')}_geometry.npz"
        np.savez_compressed(artifact, **arrays)
        rows.append(
            dict(
                pair=pair,
                method=method,
                forward_fit=info,
                reverse_fit=reverse_info,
                summary=stats,
                artifact=artifact.name,
                sha256=file_sha256(artifact),
            )
        )
        print(method, pair, stats, flush=True)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during run: {p}")
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Native RGB pixel units for derivatives/cycle.",
            "Reverse fit is separate, not a mathematical inverse.",
            "Grid step16 is not an exhaustive topology certificate.",
            "Locally regular and reciprocal mappings may still be wrong.",
        ],
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
