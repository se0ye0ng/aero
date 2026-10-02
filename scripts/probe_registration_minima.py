"""Swap only the pretrained XoFTR weights on frozen external and Anti-UAV panels."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.registration.local_consensus_warp import fit_image_warp
from aero_ir.utils.manifest import file_sha256
from scripts.analyze_registration_target_support import attribute
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_detector_free_matching import VENDOR_COMMIT, read_pair
from scripts.probe_external_registration_landmarks import (
    resize_gray,
    summarize,
    to_native,
    validate_landmarks,
)
from scripts.probe_match_geometry import fit_geometry, unique_reciprocal
from scripts.probe_registration_rgb_resolution import ignored_bytecode_paths


def read_json(path):
    return json.loads(Path(path).read_text())


def score_local(arrays, boxes, settings, policy):
    directions = {}
    sizes = [(640, 360), (640, 512)]
    for name, offset, shapes, targets in (
            ("rgb_to_ir", 0, sizes, boxes), ("ir_to_rgb", 3, sizes[::-1], boxes[::-1])):
        p, q, confidence = arrays[offset:offset+3]
        predict, info = fit_image_warp(p, q, confidence, *shapes, settings, policy)
        iou = corner_iou(predict, *targets)
        directions[name] = dict(fit=info, **attribute(p, q, info, *targets, iou))
    return dict(directions=directions, joint_box_proxy_pass=all(
        v["failure_stage"] == "box_proxy_pass" for v in directions.values()))


def save_matches(path, arrays):
    np.savez_compressed(path, **dict(zip(("points0", "points1", "confidence", "reverse0",
                                        "reverse1", "reverse_confidence"), arrays, strict=True)))
    return dict(matches_file=path.name, matches_sha256=file_sha256(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiment/registration_minima_cpu.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["box_iou_threshold"] != .6:
        raise ValueError("frozen proxy threshold is0.6")
    baseline_path = Path(config["baseline"])
    if file_sha256(baseline_path) != config["baseline_sha256"]:
        raise ValueError("baseline identity differs")
    hashes = dict(read_json(baseline_path)["input_and_source_sha256"])
    weights = Path(config["weights"])
    if (file_sha256(weights) != config["weights_sha256"]
            or weights.stat().st_size != config["weights_bytes"]):
        raise ValueError("checkpoint identity differs")
    vendor = Path(config["vendor"])
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != VENDOR_COMMIT:
        raise ValueError("unexpected inference code revision")
    ignored = ignored_bytecode_paths(subprocess.check_output([
        "git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=all"], text=True))
    if (file_sha256(config["upstream_config"]) != config["upstream_config_sha256"]
            or Path(config["upstream_config"]).read_bytes() != (
                vendor/"src/config/default.py").read_bytes()):
        raise ValueError("upstream network configuration differs")
    geometry_path = Path(config["geometry_baseline"])
    if file_sha256(geometry_path) != config["geometry_baseline_sha256"]:
        raise ValueError("global geometry baseline differs")
    extra = [args.config, Path(__file__), baseline_path, weights, geometry_path,
             Path(config["upstream_config"]), Path(config["source_report"]),
             Path(config["external_config"]), Path(config["local_config"]),
             Path("scripts/probe_registration_rgb_resolution.py"),
             Path("scripts/probe_detector_free_matching.py"),
             Path("scripts/probe_match_geometry.py"),
             Path("scripts/analyze_registration_target_support.py")]
    extra += sorted(weights.parent.glob("*.py")) + sorted(weights.parent.glob("*.json"))
    extra += [weights.parent/"LICENSE"] + sorted((vendor/"src").rglob("*.py"))
    for path in extra:
        if str(path) in hashes and hashes[str(path)] != file_sha256(path):
            raise ValueError(f"previously frozen source changed: {path}")
        hashes[str(path)] = file_sha256(path)
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError(f"changed dependency: {name}")
    settings = yaml.safe_load(Path(config["local_config"]).read_text())
    settings["maximum_controls"] = config["maximum_controls"]
    ext = yaml.safe_load(Path(config["external_config"]).read_text())
    root = Path(ext["cache"])
    manifest = read_json(root/"manifest.json")
    source = read_json(config["source_report"])
    if source["evaluated_split"] != "train" or source["validation_or_test_access"] != "none":
        raise ValueError("train panel only")
    cases = [r for r in source["rows"] if r["condition"] == "input_header_crop"]
    metadata = {r["sequence_id"]: r for r in source["inputs"]}
    if len(cases) != 16 or len({r["sequence_id"] for r in cases}) != 16:
        raise ValueError("incomplete train16 panel")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="aero-minima-bytecode-")
    sys.dont_write_bytecode = True
    model = load_matcher("xoftr", weights, vendor.resolve(), "cpu")
    started = time.monotonic()
    external_rows, anti_rows, geometry_rows, controls = [], [], [], []
    with torch.inference_mode():
        for pair in manifest["pairs"]:
            folder = root/pair
            images = [cv2.imread(str(folder/n), cv2.IMREAD_GRAYSCALE) for n in ("V.JPG", "T.JPG")]
            if any(im is None for im in images):
                raise ValueError("external image decode failed")
            small, scales = zip(*(resize_gray(im, ext["long_side"]) for im in images), strict=True)
            # Preserve the original external benchmark's thermal-first inference order.
            pt, pr, confidence, flags = infer(model, "xoftr", *small[::-1], "cpu")
            rgb, thermal = to_native(pr, scales[0]), to_native(pt, scales[1])
            path = args.out_dir/f'external_{pair.replace("/", "_")}.npz'
            np.savez_compressed(path, rgb=rgb, thermal=thermal, confidence=confidence)
            for policy in config["control_policies"]:
                predict, info = fit_image_warp(rgb, thermal, confidence,
                    images[0].shape[::-1], images[1].shape[::-1], settings, policy)
                # Reference positions are queried only after image-only fitting.
                gt_r = validate_landmarks(np.loadtxt(folder/"points_rgb.txt"), images[0].shape)
                gt_t = validate_landmarks(np.loadtxt(folder/"points_thermal.txt"), images[1].shape)
                if gt_r.shape != gt_t.shape:
                    raise ValueError("unpaired references")
                predicted = np.full_like(gt_r, np.nan) if predict is None else predict(gt_r)
                errors = np.linalg.norm(predicted-gt_t, axis=1)
                errors = np.where(np.isfinite(errors), errors, np.inf)
                external_rows.append(dict(pair=pair, policy=policy, fit=info, flags=flags,
                    matches_file=path.name, matches_sha256=file_sha256(path),
                    errors=[float(e) if np.isfinite(e) else None for e in errors],
                    summary=summarize(errors, settings["thresholds_thermal_file_pixels"])))
            print(f"external {pair} complete", flush=True)
        first_gray = None
        for index, case in enumerate(cases):
            seq = case["sequence_id"]
            images, _, observed = read_pair(Path(config["dataset_root"]), seq, 640)
            if observed != metadata[seq]:
                raise ValueError("native/decoded/resized frame identity differs")
            tops = case["header_rows"]
            gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)[top:]
                    for im, top in zip(images, tops, strict=True)]
            a, b, c, flags = infer(model, "xoftr", *gray, "cpu")
            ra, rb, rc, rflags = infer(model, "xoftr", *gray[::-1], "cpu")
            arrays = [a+[0, tops[0]], b+[0, tops[1]], c,
                      ra+[0, tops[1]], rb+[0, tops[0]], rc]
            identity = save_matches(args.out_dir/f"antiuav_{index:03d}.npz", arrays)
            boxes = []
            for name in ("visible", "infrared"):
                ann = Path(config["dataset_root"])/"train"/seq/f"{name}.json"
                x, y, w, h = read_json(ann)["gt_rect"][case["frame_index"]]
                entry = observed["inputs"][name]
                nh, nw = entry["native_shape"]
                bh, bw = entry["resized_shape"]
                boxes.append(np.array([x, y, x+w, y+h])*[bw/nw, bh/nh, bw/nw, bh/nh])
            for policy in config["control_policies"]:
                anti_rows.append(dict(sequence_id=seq, frame_index=case["frame_index"],
                    policy=policy, **identity, flags=[flags, rflags],
                    **score_local(arrays, boxes, settings, policy)))
            p, q = unique_reciprocal(arrays[0], arrays[1], arrays[3], arrays[4])
            domains = [np.array([0, top, 639, height-1]) for top, height in zip(
                tops, (360, 512), strict=True)]
            for family in config["global_models"]:
                geometry_rows.append(dict(sequence_id=seq, frame_index=case["frame_index"],
                    **fit_geometry(p, q, family,
                                   [np.array(b) for b in case["boxes_xyxy"]], domains)))
            if index == 0:
                first_gray = gray[0].copy()
                shifted = cv2.warpAffine(first_gray, np.float32([[1, 0, 8], [0, 1, -8]]),
                                        (first_gray.shape[1], first_gray.shape[0]))
                x, y, _, f = infer(model, "xoftr", first_gray, shifted, "cpu")
                e = np.linalg.norm(y-x-[8, -8], axis=1)
                controls.append(dict(kind="same_modality_known_shift", matches=len(x), flags=f,
                                     pck3=float((e <= 3).mean()) if len(e) else None))
            print(f"Anti-UAV {index+1}/16 complete", flush=True)
        x, y, c, flags = infer(model, "xoftr", first_gray, gray[1], "cpu")
        path = args.out_dir/"unrelated_pair.npz"
        np.savez_compressed(path, points0=x, points1=y, confidence=c)
        controls.append(dict(kind="unrelated_first_rgb_last_ir", matches=len(x), flags=flags,
                             matches_file=path.name, matches_sha256=file_sha256(path)))
    summary = dict(external={}, antiuav_local={}, antiuav_global={})
    for policy in config["control_policies"]:
        group = [r for r in external_rows if r["policy"] == policy]
        summary["external"][policy] = {str(t): float(np.mean([
            r["summary"]["pck_all_landmarks"][str(t)] for r in group]))
            for t in settings["thresholds_thermal_file_pixels"]}
        group = [r for r in anti_rows if r["policy"] == policy]
        summary["antiuav_local"][policy] = dict(pairs=len(group),
            joint_box_proxy_passes=sum(r["joint_box_proxy_pass"] for r in group),
            forward_fits=sum(r["directions"]["rgb_to_ir"]["fit_status"] == "fit" for r in group))
    for family in config["global_models"]:
        group = [r for r in geometry_rows if r["model"] == family]
        summary["antiuav_global"][family] = dict(pairs=len(group),
            eligible_fits=sum(r["status"] == "fit" for r in group),
            joint_box_proxy_passes=sum(r["status"] == "fit" and min(
                r["forward_box_iou_proxy"], r["inverse_box_iou_proxy"]) >= .6 for r in group))
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError(f"source changed during evaluation: {name}")
    report = dict(kind=config["kind"], qualification=config["qualification"], device="cpu",
        external_rows=external_rows, antiuav_rows=anti_rows, geometry_rows=geometry_rows,
        summary=summary, controls=controls, input_and_source_sha256=hashes,
        vendor_bytecode_ignored=ignored, bytecode_lookups_isolated=True,
        torch_version=torch.__version__, elapsed_seconds=time.monotonic()-started,
        limitations=["Checkpoint swap, not local retraining or new qualification.",
                     "Previously inspected fixed panels, not confirmatory held-out results.",
                     "External reference errors and Anti-UAV box proxies are different evidence.",
                     "Official benchmark preprocessing is replaced by our frozen baseline input.",
                     "Initial checkpoint download trusts official HTTPS, not a published digest."])
    with (args.out_dir/"report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
