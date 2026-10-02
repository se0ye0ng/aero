"""Frozen image-only XoFTR/TPS input-resolution comparison on authored landmarks."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_local_warp import fit_image_warp
from scripts.probe_external_registration_landmarks import (
    resize_gray,
    summarize,
    to_native,
    validate_landmarks,
)

BASE = Path("experiments/registration_external_landmarks_cpu_01/report.json")
LOCAL = Path("experiments/registration_external_local_warp_cpu_01/report.json")
CONFIG = Path("configs/experiment/registration_external_landmarks_cpu.yaml")
TPS = Path("configs/experiment/registration_external_local_warp_cpu.yaml")
RESOLUTIONS = (640, 1280)
REPLAY_ATOL_PX = 1e-8


def validate_replay(info, stats, errors, expected_fit, expected_stats, expected_errors, label):
    """Allow roundoff only in continuous errors, never in fit or PCK decisions."""
    if info != expected_fit:
        changed = sorted(
            k for k in info.keys() | expected_fit.keys() if info.get(k) != expected_fit.get(k)
        )
        raise ValueError(f"{label}: fitted controls/metadata differ: {changed}")
    if stats.keys() != expected_stats.keys():
        raise ValueError(f"{label}: summary fields differ")
    continuous = {"conditional_finite_median", "conditional_finite_p95"}
    for key, value in stats.items():
        expected = expected_stats[key]
        if key in continuous and value is not None and expected is not None:
            equal = (
                np.isfinite(value)
                and np.isfinite(expected)
                and abs(value - expected) <= REPLAY_ATOL_PX
            )
        else:
            equal = value == expected
        if not equal:
            raise ValueError(
                f"{label}: summary.{key} differs: actual={value!r}, expected={expected!r}"
            )
    actual = np.asarray(errors, dtype=float)
    expected = np.asarray(expected_errors, dtype=float)
    if (
        actual.shape != expected.shape
        or np.isnan(actual).any()
        or np.isnan(expected).any()
        or (actual < 0).any()
        or (expected < 0).any()
    ):
        raise ValueError(f"{label}: invalid error arrays or mismatched shapes")
    try:
        np.testing.assert_allclose(actual, expected, atol=REPLAY_ATOL_PX, rtol=0, equal_nan=False)
    except AssertionError as exc:
        raise ValueError(
            f"{label}: landmark errors differ beyond {REPLAY_ATOL_PX} px: {exc}"
        ) from exc
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0


def protocol():
    if file_sha256(BASE) != "2d74696136ca4f99efdea4e2f816b3be0742c2e9b9505a56096d8261875a4d6f":
        raise ValueError("baseline changed")
    if file_sha256(LOCAL) != "0c1d8e35cebdc67f507cf94edc3f2dd7a77782bd2cb149dda36f6394f15d86d8":
        raise ValueError("local TPS evidence changed")
    baseline, local = (json.loads(p.read_text()) for p in (BASE, LOCAL))
    hashes = baseline["source_and_weights_sha256"].copy()
    # The recorded local run pins the postprocessor and configuration.
    for p, digest in local["input_and_source_sha256"].items():
        if p in hashes and hashes[p] != digest:
            raise ValueError("conflicting frozen identities")
        hashes[p] = digest
    config = yaml.safe_load(CONFIG.read_text())
    tps = yaml.safe_load(TPS.read_text())
    root = Path(config["cache"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        file_sha256(manifest_path)
        != "0b389d397dd5c74174bc6befa2340dfe4747fd44df14b285a3dd281b8579bfb3"
    ):
        raise ValueError("reference panel changed")
    for item in manifest["files"]:
        key = str(root / item["path"])
        if key in hashes and hashes[key] != item["sha256"]:
            raise ValueError("input identity conflict")
        hashes[key] = item["sha256"]
    for p in (
        BASE,
        LOCAL,
        CONFIG,
        TPS,
        manifest_path,
        Path(__file__),
        Path("scripts/run_external_resolution_gpu.sh"),
    ):
        key, digest = str(p), file_sha256(p)
        if key in hashes and hashes[key] != digest:
            raise ValueError("changed source")
        hashes[key] = digest
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed input/source: {p}")
    return config, tps, manifest, hashes


def reference_score(root, pair, rgb_shape, thermal_shape, warp, thresholds):
    gt_rgb = validate_landmarks(np.loadtxt(root / pair / "points_rgb.txt"), rgb_shape)
    gt_thr = validate_landmarks(np.loadtxt(root / pair / "points_thermal.txt"), thermal_shape)
    if gt_rgb.shape != gt_thr.shape:
        raise ValueError("unpaired reference points")
    predicted = np.full_like(gt_rgb, np.nan) if warp is None else warp(gt_rgb)
    errors = np.linalg.norm(predicted - gt_thr, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    return errors, summarize(errors, thresholds)


def aggregate(rows):
    return [
        dict(
            long_side=size,
            pairs=len(selected),
            landmarks=sum(r["summary"]["landmarks"] for r in selected),
            macro_pck={
                str(t): float(
                    np.mean([r["summary"]["pck_all_landmarks"][str(t)] for r in selected])
                )
                for t in (1.0, 3.0, 5.0, 10.0)
            },
        )
        for size in RESOLUTIONS
        if (selected := [r for r in rows if r["long_side"] == size])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "verify"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config, tps, manifest, hashes = protocol()
    root = Path(config["cache"])
    plan = dict(
        input_and_source_sha256=hashes,
        pairs=manifest["pairs"],
        resolutions=list(RESOLUTIONS),
        method="xoftr",
        settings="Frozen local-consensus TPS and reference denominators",
        independent_reference="author-provided landmarks used only after fitting",
        qualification=False,
        antiuav_approval=False,
        generator_training_approved=False,
    )
    if args.action == "preflight":
        if args.out_dir.exists():
            raise FileExistsError(args.out_dir)
        previous = json.loads(LOCAL.read_text())
        replay_deltas = {}
        for pair in manifest["pairs"]:
            with Image.open(root / pair / "V.JPG") as im:
                rgb_size = im.size
            with Image.open(root / pair / "T.JPG") as im:
                thr_size = im.size
            matches = BASE.parent / f"xoftr_{pair.replace('/', '_')}_matches.npz"
            with np.load(matches, allow_pickle=False) as f:
                warp, info = fit_image_warp(
                    f["rgb"], f["thermal"], f["confidence"], rgb_size, thr_size, tps
                )
            errors, stats = reference_score(
                root,
                pair,
                rgb_size[::-1],
                thr_size[::-1],
                warp,
                tps["thresholds_thermal_file_pixels"],
            )
            old = next(r for r in previous["rows"] if r["pair"] == pair and r["method"] == "xoftr")
            replay_deltas[pair] = validate_replay(
                info,
                stats,
                errors,
                old["fit"],
                old["summary"],
                [np.inf if e is None else e for e in old["errors"]],
                f"CPU baseline replay {pair}",
            )
        print(
            json.dumps(
                dict(
                    ok=True,
                    inputs=len(hashes),
                    pairs=manifest["pairs"],
                    resolutions=list(RESOLUTIONS),
                    baseline_replayed_pairs=len(manifest["pairs"]),
                    replay_max_abs_error_delta_px=replay_deltas,
                    gpu_execution_pending=True,
                ),
                indent=2,
            )
        )
        return
    if args.action == "verify":
        r = json.loads((args.out_dir / "report.json").read_text())
        if (
            r["input_and_source_sha256"] != hashes
            or r["generator_training_approved"]
            or r["registration_qualified"]
        ):
            raise ValueError("provenance/approval mismatch")
        if json.loads((args.out_dir / "preflight.json").read_text()) != plan:
            raise ValueError("plan changed")
        keys = [(row["pair"], row["long_side"]) for row in r["rows"]]
        expected = {(p, s) for p in manifest["pairs"] for s in RESOLUTIONS}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("missing or duplicate cases")
        for row in r["rows"]:
            path = args.out_dir / row["artifact"]
            if (
                path.resolve().parent != args.out_dir.resolve()
                or file_sha256(path) != row["sha256"]
            ):
                raise ValueError("unsafe or changed artifact")
            with np.load(path, allow_pickle=False) as f:
                a = dict(f)
            with Image.open(root / row["pair"] / "V.JPG") as im:
                rgb_size = im.size
            with Image.open(root / row["pair"] / "T.JPG") as im:
                thr_size = im.size
            warp, info = fit_image_warp(
                a["rgb"], a["thermal"], a["confidence"], rgb_size, thr_size, tps
            )
            e, stats = reference_score(
                root,
                row["pair"],
                rgb_size[::-1],
                thr_size[::-1],
                warp,
                tps["thresholds_thermal_file_pixels"],
            )
            validate_replay(
                info,
                stats,
                e,
                row["fit"],
                row["summary"],
                a["errors"],
                f"Saved-result replay {row['pair']} cap={row['long_side']}",
            )
        if aggregate(r["rows"]) != r["aggregate"]:
            raise ValueError("aggregate changed")
        print(
            json.dumps(
                dict(
                    ok=True,
                    verified_cases=len(keys),
                    report_sha256=file_sha256(args.out_dir / "report.json"),
                    aggregate=r["aggregate"],
                    registration_qualified=False,
                ),
                indent=2,
            )
        )
        return
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this command requires the allocated GPU")
    sys.dont_write_bytecode = True
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "preflight.json").write_text(json.dumps(plan, indent=2, allow_nan=False))
    model = load_matcher(
        "xoftr", Path(config["xoftr_weights"]), Path(config["xoftr_vendor"]), "cuda"
    )
    rows = []
    for size in RESOLUTIONS:
        for pair in manifest["pairs"]:
            thermal = cv2.imread(str(root / pair / "T.JPG"), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(str(root / pair / "V.JPG"), cv2.IMREAD_GRAYSCALE)
            if thermal is None or rgb is None:
                raise ValueError("image decode failed")
            a, sa = resize_gray(thermal, size)
            b, sb = resize_gray(rgb, size)
            with torch.inference_mode():
                pt, pr, confidence, flags = infer(model, "xoftr", a, b, "cuda")
            pt, pr = to_native(pt, sa), to_native(pr, sb)
            warp, info = fit_image_warp(
                pr, pt, confidence, rgb.shape[::-1], thermal.shape[::-1], tps
            )
            errors, stats = reference_score(
                root, pair, rgb.shape, thermal.shape, warp, tps["thresholds_thermal_file_pixels"]
            )
            path = args.out_dir / f"xoftr_{size}_{pair.replace('/', '_')}.npz"
            np.savez_compressed(path, rgb=pr, thermal=pt, confidence=confidence, errors=errors)
            rows.append(
                dict(
                    pair=pair,
                    long_side=size,
                    artifact=path.name,
                    sha256=file_sha256(path),
                    matches=len(pr),
                    flags=flags,
                    fit=info,
                    summary=stats,
                    input_shapes=[list(a.shape), list(b.shape)],
                )
            )
            print(size, pair, stats, flush=True)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during run: {p}")
    result = dict(
        rows=rows,
        aggregate=aggregate(rows),
        input_and_source_sha256=hashes,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        registration_qualified=False,
        generator_training_approved=False,
        note="Four previously observed pairs; no reference-guided fitting. "
        "Landmark uncertainty is unquantified. No data redistribution or Anti-UAV approval.",
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
