"""Residual RoMa matching after trained coarse alignment; not qualification."""

import argparse
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from aero_ir.registration.prewarp import compose, field_predictor
from aero_ir.registration.roma_coordinates import rgb_image
from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.fetch_minima_roma import COMMIT, ROOT
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.verify_antiuav_tiled_matching import safe_artifact

INPUT = Path("experiments/registration_roma_prewarp_inputs_01/manifest.json")
INPUT_SHA = "84b303302cc34d588504d5fb4ad56d050724570a8201036df8cfc55ebcbff706"


def protocol():
    if file_sha256(INPUT) != INPUT_SHA:
        raise ValueError("prepared input manifest changed")
    manifest = json.loads(INPUT.read_text())
    hashes = manifest["input_and_source_sha256"].copy()
    for path, sha in hashes.items():
        if file_sha256(path) != sha:
            raise ValueError(f"source/input changed: {path}")
    for row in manifest["rows"]:
        artifact = safe_artifact(INPUT.parent, row)
        hashes[str(artifact)] = row["sha256"]
    for p in (INPUT, Path(__file__), Path("scripts/run_roma_prewarp_gpu.sh")):
        hashes[str(p)] = file_sha256(p)
    return manifest, hashes


def scores(warp, certainty, row, prepared):
    maps = split_dense(warp, certainty)
    sizes, tops, boxes = row["image_sizes"], row["header_rows"], row["boxes_xyxy"]
    crop_size = (sizes[1][0], sizes[1][1] - tops[1])
    coarse = (
        field_predictor(prepared["rgb_to_ir_field"], sizes[0], sizes[1]),
        field_predictor(prepared["ir_to_rgb_field"], sizes[1], sizes[0]),
    )
    result = {}
    for name, gated in (("confidence_ge_half", True), ("ungated_diagnostic", False)):
        residual = [
            dense_predictor(
                f, c if gated else np.ones_like(c), crop_size, crop_size, tops[1], tops[1]
            )
            for f, c in maps
        ]
        predictors = compose(
            *coarse, *residual, prepared["observed"], rgb_top=tops[0], ir_top=tops[1]
        )
        ious = [corner_iou(fn, boxes[d], boxes[1 - d]) for d, fn in enumerate(predictors)]
        result[name] = dict(iou=ious, both_ge_06=all(v is not None and v >= 0.6 for v in ious))
    return result


def control_score(warp, certainty, shape):
    field, confidence = split_dense(warp, certainty)[0]
    h, w = shape
    y, x = np.mgrid[16 : h - 16 : 16, 16 : w - 16 : 16]
    query = np.c_[x.ravel(), y.ravel()]
    error = np.linalg.norm(
        dense_predictor(field, np.ones_like(confidence), (w, h), (w, h))(query) - query - [8, -8],
        axis=1,
    )
    return dict(queries=len(query), pck3_all=float((np.isfinite(error) & (error <= 3)).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight", "run", "verify"))
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()
    manifest, hashes = protocol()
    path = args.out_dir / "report.json"
    if args.action == "preflight":
        print(json.dumps(dict(pairs=len(manifest["rows"]), identities=len(hashes), gpu_run=False)))
        return
    if args.action == "verify":
        r = json.loads(path.read_text())
        if (
            r["input_and_source_sha256"] != hashes
            or r["registration_qualified"]
            or r["generator_training_approved"]
        ):
            raise ValueError("provenance/approval differs")
        if [x["sequence_id"] for x in r["rows"]] != [x["sequence_id"] for x in manifest["rows"]]:
            raise ValueError("pair inventory differs")
        for row, original in zip(r["rows"], manifest["rows"], strict=True):
            with np.load(safe_artifact(INPUT.parent, original), allow_pickle=False) as data:
                with np.load(safe_artifact(args.out_dir, row), allow_pickle=False) as saved:
                    if scores(saved["warp"], saved["certainty"], original, data) != row["scores"]:
                        raise ValueError("saved score differs")
        c = r["known_shift"]
        with np.load(safe_artifact(args.out_dir, c), allow_pickle=False) as saved:
            if control_score(saved["warp"], saved["certainty"], c["shape"]) != c["score"]:
                raise ValueError("control differs")
        print(json.dumps(dict(ok=True, pairs=len(r["rows"]), neural_inference_replayed=False)))
        return
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    sys.path.insert(0, str((ROOT / f"RoMa_minima-{COMMIT}").resolve()))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    error = RuntimeError("implicit network access prohibited")
    rows = []
    with (
        patch.object(socket.socket, "connect", side_effect=error),
        patch.object(torch.hub, "load_state_dict_from_url", side_effect=error),
        patch.object(torch.hub, "load", side_effect=error),
        torch.inference_mode(),
    ):
        from romatch import roma_outdoor

        model = roma_outdoor(
            device="cuda",
            amp_dtype=torch.float32,
            weights=torch.load(ROOT / "minima_roma.pth", map_location="cpu", weights_only=True),
            dinov2_weights=torch.load(
                ROOT / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True
            ),
        )
        model.eval()
        for module in model.modules():
            if hasattr(module, "amp"):
                module.amp = False

        def infer(images, filename):
            w, c = model.match(*(rgb_image(im) for im in images), device="cuda")
            w, c = w.float().cpu().numpy(), c.float().cpu().numpy()
            split_dense(w, c)
            target = args.out_dir / filename
            np.savez_compressed(target, warp=w, certainty=c)
            return w, c, dict(artifact=filename, sha256=file_sha256(target))

        first = None
        for i, original in enumerate(manifest["rows"]):
            with np.load(safe_artifact(INPUT.parent, original), allow_pickle=False) as data:
                images = [data["warped_rgb"], data["infrared"]]
                if first is None:
                    first = images[0].copy()
                w, c, artifact = infer(images, f"{i:03d}_residual.npz")
                result = scores(w, c, original, data)
            rows.append(dict(sequence_id=original["sequence_id"], scores=result, **artifact))
            print(f"{i + 1}/16 {original['sequence_id']}: {result}", flush=True)
        h, w = first.shape[:2]
        shifted = cv2.warpAffine(first, np.array([[1, 0, 8], [0, 1, -8]], float), (w, h))
        cw, cc, control = infer([first, shifted], "known_shift.npz")
        control.update(shape=[h, w], score=control_score(cw, cc, (h, w)))
    if protocol()[1] != hashes:
        raise ValueError("source/input changed during inference")
    result = dict(
        rows=rows,
        known_shift=control,
        input_and_source_sha256=hashes,
        coarse_box_proxy_both_ge_06=manifest["coarse_box_proxy_both_ge_06"],
        registration_qualified=False,
        generator_training_approved=False,
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        limitations=[
            "Train16 exploration; boxes score only, not pixel GT.",
            "Coarse model was trained on these sequences.",
            "Confidence-disabled branch is diagnostic only.",
        ],
    )
    with path.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
