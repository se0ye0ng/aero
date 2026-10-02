"""Frozen same-scene unused-pair confirmation, without GT-guided control selection."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial import QhullError

from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_boundary_controls import accepted_matches, augment_boundary
from scripts.probe_external_local_warp import fit_image_warp
from scripts.probe_external_orientation_repair import selected_tps
from scripts.probe_external_registration_landmarks import resize_gray, to_native
from scripts.probe_external_resolution import protocol, reference_score

CONFIG = Path("configs/experiment/registration_external_boundary_confirmation.yaml")
CONFIG_SHA = "b245cacb97d7647794161eaff6d9260f16c08e47586d61c14fc6dd38cfac4a72"
MANIFEST_SHA = "c37bfa797fe537552675ff2917b87267be7deaae4dfc61351a754dc283d7dab2"


def aggregate(rows):
    result = {}
    for subset in ("all12", "without_development_duplicate_labels"):
        selected = rows if subset == "all12" else [r for r in rows if not r["duplicate_labels"]]
        result[subset] = dict(pairs=len(selected), variants={})
        for variant in ("baseline", "boundary"):
            summaries = [r[variant] for r in selected]
            result[subset]["variants"][variant] = dict(
                landmarks=sum(s["landmarks"] for s in summaries),
                unavailable=sum(s["unprojectable"] for s in summaries),
                macro_pck={
                    str(t): float(np.mean([s["pck_all_landmarks"][str(t)] for s in summaries]))
                    for t in (1.0, 3.0, 5.0, 10.0)
                },
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("confirmation policy changed")
    c = yaml.safe_load(CONFIG.read_text())
    _, settings, _, hashes = protocol()
    for path, expected in (
        (c["candidate_source"], c["candidate_sha256"]),
        (c["development_report"], c["development_report_sha256"]),
    ):
        if file_sha256(path) != expected:
            raise ValueError("candidate or development evidence changed")
    root = Path(c["cache"])
    manifest_path = root / "manifest.json"
    if file_sha256(manifest_path) != MANIFEST_SHA:
        raise ValueError("confirmation input manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != CONFIG_SHA:
        raise ValueError("download policy differs")
    for f in manifest["files"]:
        hashes[str(root / f["path"])] = f["sha256"]
    for p in (
        CONFIG,
        manifest_path,
        Path(__file__),
        Path(c["candidate_source"]),
        Path(c["development_report"]),
        Path("scripts/probe_external_orientation_repair.py"),
        Path("scripts/analyze_external_control_topology.py"),
    ):
        hashes[str(p)] = file_sha256(p)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed input: {p}")
    pairs = [f"{scene}/{i}" for scene in c["scenes"] for i in c["confirmation_sample_ids"]]
    if len(pairs) != 12 or not set(pairs).issubset(manifest["pairs"]):
        raise ValueError("incomplete panel")
    files = {f["path"]: f["sha256"] for f in manifest["files"]}
    duplicates = {}
    for pair in pairs:
        scene = pair.split("/")[0]
        duplicates[pair] = all(
            files[f"{pair}/{name}"] == files[f"{scene}/1/{name}"]
            for name in ("points_rgb.txt", "points_thermal.txt")
        )
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="aero-boundary-confirm-bytecode-")
    sys.dont_write_bytecode = True
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = dict(
        pairs=pairs,
        duplicate_labels=duplicates,
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(plan, indent=2, allow_nan=False))
    model = load_matcher(
        "xoftr", Path(c["xoftr_weights"]), Path(c["xoftr_vendor"]).resolve(), "cpu"
    )
    assert model.fine_matching.denser is False
    rows = []
    with torch.inference_mode():
        for pair in pairs:
            rgb = cv2.imread(str(root / pair / "V.JPG"), cv2.IMREAD_GRAYSCALE)
            thr = cv2.imread(str(root / pair / "T.JPG"), cv2.IMREAD_GRAYSCALE)
            if rgb is None or thr is None:
                raise ValueError("image decode failed")
            a, sa = resize_gray(thr, 640)
            b, sb = resize_gray(rgb, 640)
            pt, pr, confidence, flags = infer(model, "xoftr", a, b, "cpu")
            pt, pr = to_native(pt, sa), to_native(pr, sb)
            sizes = (rgb.shape[::-1], thr.shape[::-1])
            original, info = fit_image_warp(pr, pt, confidence, *sizes, settings)
            candidate, ids = None, np.array([], dtype=int)
            status = info["status"]
            if original is not None:
                accepted, grid = accepted_matches(pr, pt, confidence, *sizes, settings)
                np.testing.assert_array_equal(grid, info["control_match_indices"])
                assert len(accepted) == info["accepted_matches"]
                try:
                    ids = augment_boundary(pr, accepted, grid)
                    candidate = selected_tps(pr[ids], pt[ids], *sizes, settings["tps_smoothing"])
                except (ValueError, QhullError, np.linalg.LinAlgError):
                    status = "boundary_fit_failed"
            # References are read only after both transforms have been fitted.
            errors, baseline = reference_score(
                root, pair, rgb.shape, thr.shape, original, c["thresholds_thermal_file_pixels"]
            )
            after, boundary = reference_score(
                root, pair, rgb.shape, thr.shape, candidate, c["thresholds_thermal_file_pixels"]
            )
            path = args.out_dir / f"{pair.replace('/', '_')}.npz"
            np.savez_compressed(
                path,
                rgb=pr,
                thermal=pt,
                confidence=confidence,
                original_errors=errors,
                boundary_errors=after,
                expanded_controls=ids,
            )
            row = dict(
                pair=pair,
                duplicate_labels=duplicates[pair],
                baseline=baseline,
                boundary=boundary,
                original_fit=info,
                candidate_status=status,
                flags=flags,
                artifact=path.name,
                sha256=file_sha256(path),
            )
            rows.append(row)
            print(
                pair,
                baseline["pck_all_landmarks"]["3.0"],
                boundary["pck_all_landmarks"]["3.0"],
                flush=True,
            )
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed: {p}")
    report = dict(
        rows=rows,
        aggregate=aggregate(rows),
        input_and_source_sha256=hashes,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Same scenes, unused pair IDs; not unseen-scene generalization.",
            "Duplicate reference labels explicitly retained and stratified.",
            "Author landmark uncertainty unquantified; no Anti-UAV approval.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
