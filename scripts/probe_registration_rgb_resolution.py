"""Frozen XoFTR with higher RGB resolution and unchanged native IR, on CPU."""
from __future__ import annotations

import argparse
import hashlib
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
from scripts.audit_antiuav300_dense_registration import _read_at
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_detector_free_matching import VENDOR_COMMIT, WEIGHT_HASHES


def prepare_input(native_bgr, base_hw, base_top, width):
    """Same interpolation/color order as frozen baseline; common header support."""
    bh, bw = base_hw
    if width % bw or width < bw:
        raise ValueError("resolution must be an integer multiple of the baseline width")
    factor = width // bw
    height, top = bh*factor, base_top*factor
    if width % 8 or height % 8 or top % 8 or not 0 <= top < height:
        raise ValueError("invalid eight-pixel-aligned input/header dimensions")
    rgb = cv2.cvtColor(cv2.resize(native_bgr, (width, height)), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray[top:], dict(scale=factor, top=top, full_shape=[height, width])


def restore_to_base(points, transform):
    """Restore cropped point centers, unlike box edges which use pure scaling."""
    p = np.asarray(points, dtype=np.float64)
    return (p + [0, transform["top"]] + .5)/transform["scale"]-.5


def read_json(path):
    return json.loads(Path(path).read_text())


def ignored_bytecode_paths(porcelain):
    """Allow only untracked cache files; tracked edits and other extras fail."""
    ignored = []
    for row in porcelain.splitlines():
        path = Path(row[3:])
        if (not row.startswith("?? ") or path.parent.name != "__pycache__"
                or path.suffix != ".pyc"):
            raise ValueError(f"vendor code is dirty: {row}")
        ignored.append(str(path))
    return ignored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/experiment/registration_rgb_resolution_cpu.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    baseline_path = Path(config["baseline"])
    if file_sha256(baseline_path) != config["baseline_sha256"]:
        raise ValueError("unexpected frozen baseline")
    baseline = read_json(baseline_path)
    hashes = dict(baseline["input_and_source_sha256"])
    extra_sources = [args.config, Path(__file__), baseline_path,
                     Path("src/aero_ir/registration/detector_free.py"),
                     Path("scripts/analyze_registration_target_support.py"),
                     Path("scripts/audit_antiuav300_dense_registration.py"),
                     Path("scripts/probe_detector_free_matching.py")]
    for path in extra_sources:
        if str(path) in hashes and hashes[str(path)] != file_sha256(path):
            raise ValueError(f"baseline source changed: {path}")
        hashes[str(path)] = file_sha256(path)
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError(f"changed baseline dependency: {name}")
    source_path = Path(config["source_report"])
    if str(source_path) not in hashes or config["local_config"] not in hashes:
        raise ValueError("unverified configuration or source panel")
    source = read_json(source_path)
    if source["evaluated_split"] != "train" or source["validation_or_test_access"] != "none":
        raise ValueError("only frozen train panel is allowed")
    cases = [r for r in source["rows"] if r["condition"] == "input_header_crop"]
    if len(cases) != 16 or len({r["sequence_id"] for r in cases}) != 16:
        raise ValueError("incomplete fixed16 panel")
    metadata = {r["sequence_id"]: r["inputs"] for r in source["inputs"]}
    settings = yaml.safe_load(Path(config["local_config"]).read_text())
    settings["maximum_controls"] = config["maximum_controls"]
    vendor = Path(config["external_root"])/"XoFTR"
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"],
                                     text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain",
                                     "--untracked-files=all"], text=True)
    ignored_caches = ignored_bytecode_paths(dirty)
    weights = Path(config["external_root"])/"weights_xoftr_640.ckpt"
    if commit != VENDOR_COMMIT or file_sha256(weights) != WEIGHT_HASHES["xoftr"]:
        raise ValueError("vendor or weights differ")
    hashes[str(weights)] = WEIGHT_HASHES["xoftr"]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    # Do not execute or delete preexisting vendor bytecode. Redirect all future
    # bytecode lookups to a fresh empty directory and disable bytecode writes.
    sys.pycache_prefix = tempfile.mkdtemp(prefix="aero-xoftr-bytecode-")
    sys.dont_write_bytecode = True
    model = load_matcher("xoftr", weights, vendor.resolve(), "cpu")
    rows, controls, decoded = [], [], []
    start = time.monotonic()
    first_rgb = None
    with torch.inference_mode():
        for index, case in enumerate(cases):
            seq, frame = case["sequence_id"], case["frame_index"]
            inputs, transforms, native_images = [], [], []
            for modality, name in enumerate(("visible", "infrared")):
                path = Path(config["dataset_root"])/"train"/seq/f"{name}.mp4"
                capture = cv2.VideoCapture(str(path))
                try:
                    native = _read_at(capture, frame, path)
                finally:
                    capture.release()
                entry = metadata[seq][name]
                digest = hashlib.sha256(native.tobytes()).hexdigest()
                if list(native.shape[:2]) != entry["native_shape"] or digest != entry[
                        "decoded_native_sha256"]:
                    raise ValueError("decoded native frame differs from frozen panel")
                native_images.append(native)
                base_hw = entry["resized_shape"]
                width = config["rgb_width"] if modality == 0 else base_hw[1]
                gray, transform = prepare_input(native, base_hw, case["header_rows"][modality],
                                                width)
                inputs.append(gray)
                transforms.append(transform)
                decoded.append(dict(sequence_id=seq, frame_index=frame, modality=name,
                                    native_sha256=digest, input_shape=list(gray.shape),
                                    input_sha256=hashlib.sha256(gray.tobytes()).hexdigest()))
            match_path = source_path.parent/case["matches_file"]
            if str(match_path) not in hashes:
                raise ValueError("unverified baseline matches")
            with np.load(match_path, allow_pickle=False) as data:
                base_arrays = [data[k] for k in ("points0", "points1", "confidence",
                                               "reverse0", "reverse1", "reverse_confidence")]
            if index == 0:
                low, low_transform = prepare_input(native_images[0], (360, 640),
                                                    case["header_rows"][0], 640)
                for a, b, ta, tb, offset in ((low, inputs[1], low_transform, transforms[1], 0),
                                           (inputs[1], low, transforms[1], low_transform, 3)):
                    x, y, c, _ = infer(model, "xoftr", a, b, "cpu")
                    actual = [restore_to_base(x, ta), restore_to_base(y, tb), c]
                    for new, old in zip(actual, base_arrays[offset:offset+3], strict=True):
                        if new.shape != old.shape or not np.allclose(new, old, atol=1e-4, rtol=0):
                            raise ValueError("same-resolution inference replay differs")
                print("first-pair bidirectional 640-input inference replay passed", flush=True)
            x, y, c, flags = infer(model, "xoftr", *inputs, "cpu")
            rx, ry, rc, rflags = infer(model, "xoftr", *inputs[::-1], "cpu")
            arrays = [restore_to_base(x, transforms[0]), restore_to_base(y, transforms[1]), c,
                      restore_to_base(rx, transforms[1]), restore_to_base(ry, transforms[0]), rc]
            path = args.out_dir/f"{index:03d}_rgb1280_ir640.npz"
            np.savez_compressed(path, **dict(zip(("points0", "points1", "confidence",
                "reverse0", "reverse1", "reverse_confidence"), arrays, strict=True)))
            # Read annotation boxes only after inference; no GT crop/match selection.
            boxes = []
            for name in ("visible", "infrared"):
                ann = Path(config["dataset_root"])/"train"/seq/f"{name}.json"
                if str(ann) not in hashes:
                    raise ValueError("unverified annotation")
                x0, y0, w, h = read_json(ann)["gt_rect"][frame]
                nh, nw = metadata[seq][name]["native_shape"]
                bh, bw = metadata[seq][name]["resized_shape"]
                boxes.append(np.array([x0, y0, x0+w, y0+h])*[bw/nw, bh/nh, bw/nw, bh/nh])
            sizes = [(640, 360), (640, 512)]
            for policy in config["control_policies"]:
                directions = {}
                for name, offset, ordered_sizes, ordered_boxes in (
                        ("rgb_to_ir", 0, sizes, boxes), ("ir_to_rgb", 3, sizes[::-1], boxes[::-1])):
                    p, q, confidence = arrays[offset:offset+3]
                    predict, info = fit_image_warp(p, q, confidence, *ordered_sizes, settings,
                                                  policy)
                    iou = corner_iou(predict, *ordered_boxes)
                    directions[name] = dict(fit=info, **attribute(p, q, info, *ordered_boxes, iou,
                                                                config["box_iou_threshold"]))
                rows.append(dict(sequence_id=seq, frame_index=frame, policy=policy,
                                 directions=directions, matches_file=path.name,
                                 matches_sha256=file_sha256(path), flags=[flags, rflags],
                                 joint_box_proxy_pass=all(d["failure_stage"] == "box_proxy_pass"
                                                          for d in directions.values())))
            if index == 0:
                first_rgb = inputs[0].copy()
                shifted = cv2.warpAffine(first_rgb, np.float32([[1, 0, 8], [0, 1, -8]]),
                                        (first_rgb.shape[1], first_rgb.shape[0]))
                x, y, _, flags = infer(model, "xoftr", first_rgb, shifted, "cpu")
                errors = np.linalg.norm(y-x-[8, -8], axis=1)
                controls.append(dict(kind="same_modality_known_shift_not_crossmodal_accuracy",
                                     matches=len(x), flags=flags,
                                     pck3=float((errors <= 3).mean()) if len(x) else None))
            print(f"{index+1}/16 RGB1280 pair complete ({time.monotonic()-start:.1f}s)", flush=True)
        x, y, c, flags = infer(model, "xoftr", first_rgb, inputs[1], "cpu")
        negative_path = args.out_dir/"unrelated_pair.npz"
        np.savez_compressed(negative_path, points0=x, points1=y, confidence=c)
        controls.append(dict(kind="unrelated_first_rgb_last_ir", matches=len(x), flags=flags,
                             matches_file=negative_path.name,
                             matches_sha256=file_sha256(negative_path)))
    summary = {}
    for policy in config["control_policies"]:
        selected = [r for r in rows if r["policy"] == policy]
        summary[policy] = dict(
            pairs=len(selected),
            joint_box_proxy_passes=sum(r["joint_box_proxy_pass"] for r in selected))
        for direction in ("rgb_to_ir", "ir_to_rgb"):
            values = [r["directions"][direction] for r in selected]
            summary[policy][direction] = dict(
                fits=sum(v["fit_status"] == "fit" for v in values),
                available_corner_iou=sum(v["corner_iou"] is not None for v in values),
                directional_box_passes=sum(v["failure_stage"] == "box_proxy_pass" for v in values),
                pairs_with_both_box_matches=sum(v["both_box_matches"] > 0 for v in values))
    for name, digest in hashes.items():
        if file_sha256(name) != digest:
            raise ValueError(f"input changed during run: {name}")
    report = dict(kind=config["kind"], qualification=config["qualification"], device="cpu",
                  evaluated_split="train", validation_or_test_access="none", rows=rows,
                  summary=summary, controls=controls, decoded=decoded, vendor_commit=commit,
                  torch_version=torch.__version__, opencv_version=cv2.__version__,
                  vendor_untracked_bytecode_ignored=ignored_caches,
                  vendor_bytecode_reads_isolated=True,
                  first_pair_baseline_inference_replayed=True, input_and_source_sha256=hashes,
                  elapsed_seconds=time.monotonic()-start,
                  limitations=["Known shifts and box routing do not establish physical accuracy.",
                               "RGB resolution changes feature scale as well as retained detail.",
                               "Central HUD remains; top header support is unchanged.",
                               "Only one frozen model and16 previously examined train pairs."])
    with (args.out_dir/"report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
