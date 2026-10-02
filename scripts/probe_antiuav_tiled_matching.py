"""Fixed train16 XoFTR annotation-free tiled matching comparison, GPU diagnostic."""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.registration.tiled_matching import match_tiles
from aero_ir.utils.manifest import file_sha256
from scripts.audit_antiuav300_dense_registration import _read_at
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_external_local_warp import fit_image_warp
from scripts.probe_registration_rgb_resolution import prepare_input, restore_to_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config_path = Path("configs/experiment/registration_rgb_resolution_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    prior_path = Path(config["baseline"])
    if file_sha256(prior_path) != config["baseline_sha256"]:
        raise ValueError("baseline changed")
    hashes = json.loads(prior_path.read_text())["input_and_source_sha256"].copy()
    source_path = Path(config["source_report"])
    if str(source_path) not in hashes:
        raise ValueError("unverified panel")
    source = json.loads(source_path.read_text())
    if source["evaluated_split"] != "train" or source["validation_or_test_access"] != "none":
        raise ValueError("train only")
    cases = [r for r in source["rows"] if r["condition"] == "input_header_crop"]
    if len(cases) != 16 or len({r["sequence_id"] for r in cases}) != 16:
        raise ValueError("incomplete panel")
    metadata = {r["sequence_id"]: r["inputs"] for r in source["inputs"]}
    settings = yaml.safe_load(Path(config["local_config"]).read_text())
    vendor = Path(config["external_root"]) / "XoFTR"
    weights = Path(config["external_root"]) / "weights_xoftr_640.ckpt"
    files = [
        config_path,
        prior_path,
        Path(__file__),
        weights,
        Path("scripts/probe_registration_rgb_resolution.py"),
        Path("scripts/probe_antiuav_local_warp.py"),
        Path("scripts/probe_external_local_warp.py"),
        Path("scripts/audit_antiuav300_dense_registration.py"),
        Path("src/aero_ir/registration/detector_free.py"),
        Path("src/aero_ir/registration/tiled_matching.py"),
        Path("scripts/run_antiuav_tiled_matching_gpu.sh"),
    ]
    files += list((vendor / "src").rglob("*.py"))
    for p in files:
        digest = file_sha256(p)
        if str(p) in hashes and hashes[str(p)] != digest:
            raise ValueError(f"changed frozen dependency: {p}")
        hashes[str(p)] = digest
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed: {p}")
    if args.preflight:
        print(
            json.dumps(
                dict(ok=True, pairs=len(cases), inputs=len(hashes), tile_pairs_per_direction=81)
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="aero-dense-fine-bytecode-")
    sys.dont_write_bytecode = True
    args.out_dir.mkdir(parents=True, exist_ok=False)
    model = load_matcher("xoftr", weights, vendor.resolve(), "cuda")
    model.fine_matching.denser = False

    def pair_infer(a, b):
        return infer(model, "xoftr", a, b, "cuda")

    rows, controls = [], []
    first_rgb = None
    with torch.inference_mode():
        for index, case in enumerate(cases):
            seq, frame = case["sequence_id"], case["frame_index"]
            inputs, transforms, boxes, sizes = [], [], [], []
            for m, name in enumerate(("visible", "infrared")):
                path = Path(config["dataset_root"]) / "train" / seq / f"{name}.mp4"
                cap = cv2.VideoCapture(str(path))
                try:
                    native = _read_at(cap, frame, path)
                finally:
                    cap.release()
                entry = metadata[seq][name]
                if (
                    list(native.shape[:2]) != entry["native_shape"]
                    or hashlib.sha256(native.tobytes()).hexdigest()
                    != entry["decoded_native_sha256"]
                ):
                    raise ValueError("native input changed")
                gray, transform = prepare_input(
                    native,
                    entry["resized_shape"],
                    case["header_rows"][m],
                    entry["resized_shape"][1],
                )
                inputs.append(gray)
                transforms.append(transform)
                sizes.append(tuple(entry["resized_shape"][::-1]))
            match_path = source_path.parent / case["matches_file"]
            if str(match_path) not in hashes:
                raise ValueError("unverified baseline matches")
            with np.load(match_path, allow_pickle=False) as f:
                baseline = [
                    f[k]
                    for k in (
                        "points0",
                        "points1",
                        "confidence",
                        "reverse0",
                        "reverse1",
                        "reverse_confidence",
                    )
                ]
            arrays, flags = [], []
            for direction in (0, 1):
                images = inputs if direction == 0 else inputs[::-1]
                tf = transforms if direction == 0 else transforms[::-1]
                x, y, c, base_flag = pair_infer(*images)
                baseline[direction * 3 : direction * 3 + 3] = [
                    restore_to_base(x, tf[0]),
                    restore_to_base(y, tf[1]),
                    c,
                ]
                x, y, c, flag = match_tiles(
                    pair_infer,
                    *images,
                    progress=lambda n, total, index=index, direction=direction: print(
                        f"pair {index + 1}/16 direction {direction}: {n}/{total} tile pairs",
                        flush=True,
                    ),
                )
                flag["baseline_flags"] = base_flag
                arrays.extend([restore_to_base(x, tf[0]), restore_to_base(y, tf[1]), c])
                flags.append(flag)
            # Annotation access only after image matching; no GT crop or filtering.
            for name in ("visible", "infrared"):
                ann = Path(config["dataset_root"]) / "train" / seq / f"{name}.json"
                if str(ann) not in hashes:
                    raise ValueError("unverified annotation")
                x, y, w, h = json.loads(ann.read_text())["gt_rect"][frame]
                nh, nw = metadata[seq][name]["native_shape"]
                rh, rw = metadata[seq][name]["resized_shape"]
                boxes.append(np.array([x, y, x + w, y + h]) * [rw / nw, rh / nh, rw / nw, rh / nh])
            np.testing.assert_allclose(boxes, case["boxes_xyxy"], atol=1e-4, rtol=0)
            scores = {}
            for variant, values in [("baseline", baseline), ("tiled", arrays)]:
                direction_scores = []
                for d in (0, 1):
                    warp, info = fit_image_warp(
                        *values[d * 3 : d * 3 + 3], *(sizes if d == 0 else sizes[::-1]), settings
                    )
                    iou = corner_iou(warp, *(boxes if d == 0 else boxes[::-1]))
                    direction_scores.append(dict(matches=len(values[d * 3]), fit=info, iou=iou))
                scores[variant] = dict(
                    directions=direction_scores,
                    joint_pass=all(
                        x["iou"] is not None and x["iou"] >= config["box_iou_threshold"]
                        for x in direction_scores
                    ),
                )
            artifact = args.out_dir / f"{index:03d}_matches.npz"
            np.savez_compressed(
                artifact,
                **dict(
                    zip(
                        (
                            "baseline_points0",
                            "baseline_points1",
                            "baseline_confidence",
                            "baseline_reverse0",
                            "baseline_reverse1",
                            "baseline_reverse_confidence",
                        ),
                        baseline,
                        strict=True,
                    )
                ),
                **dict(
                    zip(
                        (
                            "points0",
                            "points1",
                            "confidence",
                            "reverse0",
                            "reverse1",
                            "reverse_confidence",
                        ),
                        arrays,
                        strict=True,
                    )
                ),
            )
            rows.append(
                dict(
                    sequence_id=seq,
                    frame_index=frame,
                    scores=scores,
                    boxes_xyxy=[b.tolist() for b in boxes],
                    sizes=sizes,
                    flags=flags,
                    artifact=artifact.name,
                    sha256=file_sha256(artifact),
                )
            )
            if index == 0:
                first_rgb = inputs[0].copy()
                shifted = cv2.warpAffine(
                    first_rgb, np.float32([[1, 0, 8], [0, 1, -8]]), first_rgb.shape[::-1]
                )
                x, y, c, flag = match_tiles(pair_infer, first_rgb, shifted)
                control_path = args.out_dir / "known_shift.npz"
                np.savez_compressed(control_path, points0=x, points1=y, confidence=c)
                error = np.linalg.norm(y - x - [8, -8], axis=1)
                controls.append(
                    dict(
                        kind="same_modality_known_shift",
                        artifact=control_path.name,
                        sha256=file_sha256(control_path),
                        matches=len(x),
                        pck3=float(np.mean(error <= 3)) if len(x) else None,
                        flags=flag,
                    )
                )
            print(
                f"{index + 1}/16 complete: " + str({k: v["joint_pass"] for k, v in scores.items()}),
                flush=True,
            )
        x, y, c, flag = match_tiles(pair_infer, first_rgb, inputs[1])
        negative = args.out_dir / "unrelated.npz"
        np.savez_compressed(negative, points0=x, points1=y, confidence=c)
        controls.append(
            dict(
                kind="unrelated_pair",
                matches=len(x),
                flags=flag,
                artifact=negative.name,
                sha256=file_sha256(negative),
            )
        )
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during run: {p}")
    result = dict(
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        rows=rows,
        controls=controls,
        input_and_source_sha256=hashes,
        summary={k: sum(r["scores"][k]["joint_pass"] for r in rows) for k in ("baseline", "tiled")},
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Train16 box proxy only; unchanged confidence and geometry thresholds.",
            "Tiling changes feature scale and context, not native image information. "
            "All 81 tile pairs retained.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
