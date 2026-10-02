"""Fixed train16 RoMa dense inference and controls; never automatic qualification."""

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from aero_ir.registration.roma_coordinates import rgb_image
from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.audit_antiuav300_dense_registration import _read_at
from scripts.fetch_minima_roma import COMMIT, ROOT
from scripts.probe_antiuav_local_warp import corner_iou


def protocol():
    runtime = Path("experiments/minima_roma_runtime_cpu_01.json")
    panel = Path("experiments/xoftr_overlay_train16_01/report.json")
    for p, expected in (
        (runtime, "b785198ef9a83e36ccbce627b535682861ed18fc161887e0575c4acfba172793"),
        (panel, "09081366f56d022c7414b83b416d81837bafd37435e88bf6e554d9df2ecd18ef"),
    ):
        if file_sha256(p) != expected:
            raise ValueError(f"frozen input changed: {p}")
    hashes = json.loads(runtime.read_text())["input_and_source_sha256"].copy()
    for p in (
        runtime,
        panel,
        Path(__file__),
        Path("scripts/fetch_minima_roma.py"),
        Path("src/aero_ir/registration/roma_coordinates.py"),
        Path("src/aero_ir/registration/roma_dense.py"),
        Path("scripts/audit_antiuav300_dense_registration.py"),
        Path("scripts/probe_antiuav_local_warp.py"),
        Path("scripts/run_minima_roma_gpu.sh"),
    ):
        hashes[str(p)] = file_sha256(p)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"source changed: {p}")
    source = json.loads(panel.read_text())
    if source["evaluated_split"] != "train" or source["validation_or_test_access"] != "none":
        raise ValueError("train only")
    cases = [r for r in source["rows"] if r["condition"] == "input_header_crop"]
    if len(cases) != 16 or len({r["sequence_id"] for r in cases}) != 16:
        raise ValueError("incomplete fixed panel")
    metadata = {r["sequence_id"]: r["inputs"] for r in source["inputs"]}
    return hashes, cases, metadata


def read_images(case, metadata, root):
    images = []
    for m, name in enumerate(("visible", "infrared")):
        path = root / "train" / case["sequence_id"] / f"{name}.mp4"
        cap = cv2.VideoCapture(str(path))
        try:
            native = _read_at(cap, case["frame_index"], path)
        finally:
            cap.release()
        entry = metadata[case["sequence_id"]][name]
        if hashlib.sha256(native.tobytes()).hexdigest() != entry["decoded_native_sha256"]:
            raise ValueError("central video frame changed")
        h, w = entry["resized_shape"]
        image = cv2.cvtColor(cv2.resize(native, (w, h)), cv2.COLOR_BGR2RGB)
        images.append(image[case["header_rows"][m] :])
    return images


def evaluate(warp, certainty, images, tops, boxes):
    maps = split_dense(warp, certainty)
    sizes = [tuple(im.shape[1::-1]) for im in images]
    predictors = [
        dense_predictor(*maps[d], sizes[d], sizes[1 - d], tops[d], tops[1 - d]) for d in range(2)
    ]
    ious = [corner_iou(predictors[d], boxes[d], boxes[1 - d]) for d in range(2)]
    return dict(
        iou=ious,
        joint_pass=all(v is not None and v >= 0.6 for v in ious),
        certainty_ge_half_fraction=[float((c >= 0.5).mean()) for _, c in maps],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300")
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    hashes, cases, metadata = protocol()
    for case in cases:
        for name in ("visible", "infrared"):
            if not (args.root / "train" / case["sequence_id"] / f"{name}.mp4").is_file():
                raise FileNotFoundError(case["sequence_id"])
    if args.preflight:
        print(json.dumps(dict(ok=True, pairs=16, input_identities=len(hashes), gpu_run=False)))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cv2.setNumThreads(1)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((ROOT / f"RoMa_minima-{COMMIT}").resolve()))
    error = RuntimeError("implicit model download/network forbidden")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows, controls = [], []
    with (
        patch.object(torch.hub, "load_state_dict_from_url", side_effect=error),
        patch.object(torch.hub, "load", side_effect=error),
        patch.object(socket.socket, "connect", side_effect=error),
        torch.inference_mode(),
    ):
        from romatch import roma_outdoor

        model = roma_outdoor(
            device="cuda",
            weights=torch.load(ROOT / "minima_roma.pth", map_location="cpu", weights_only=True),
            dinov2_weights=torch.load(
                ROOT / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True
            ),
            amp_dtype=torch.float32,
        )
        model.eval()
        # The constructor sets some decoder defaults separately; disable their AMP too.
        for module in model.modules():
            if hasattr(module, "amp"):
                module.amp = False

        def infer_save(images, name):
            warp, certainty = model.match(*(rgb_image(im) for im in images), device="cuda")
            w, c = warp.float().cpu().numpy(), certainty.float().cpu().numpy()
            split_dense(w, c)
            path = args.out_dir / name
            np.savez_compressed(path, warp=w, certainty=c)
            return w, c, dict(artifact=name, sha256=file_sha256(path))

        first = last = None
        for i, case in enumerate(cases):
            images = read_images(case, metadata, args.root)
            if first is None:
                first = images[0].copy()
            last = images[1]
            w, c, artifact = infer_save(images, f"{i:03d}_dense.npz")
            scores = evaluate(w, c, images, case["header_rows"], case["boxes_xyxy"])
            rows.append(
                dict(
                    sequence_id=case["sequence_id"],
                    frame_index=case["frame_index"],
                    crop_shapes=[list(im.shape[:2]) for im in images],
                    header_rows=case["header_rows"],
                    boxes_xyxy=case["boxes_xyxy"],
                    scores=scores,
                    **artifact,
                )
            )
            print(f"pair {i + 1}/16 {case['sequence_id']}: {scores}", flush=True)
        h, width = first.shape[:2]
        shifted = cv2.warpAffine(first, np.array([[1, 0, 8], [0, 1, -8]], dtype=float), (width, h))
        for name, images in (("known_shift", [first, shifted]), ("unrelated", [first, last])):
            w, c, artifact = infer_save(images, name + ".npz")
            maps = split_dense(w, c)
            control = dict(
                kind=name,
                **artifact,
                certainty_ge_half_fraction=[float((cc >= 0.5).mean()) for _, cc in maps],
            )
            if name == "known_shift":
                yy, xx = np.mgrid[16 : h - 16 : 16, 16 : width - 16 : 16]
                query = np.c_[xx.ravel(), yy.ravel()]
                pred = dense_predictor(*maps[0], (width, h), (width, h))(query)
                error_px = np.linalg.norm(pred - query - [8, -8], axis=1)
                control.update(
                    queries=len(query),
                    supported=int(np.isfinite(error_px).sum()),
                    pck3_all=float(np.mean(error_px <= 3)),
                )
            controls.append(control)
            print(control, flush=True)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during inference: {p}")
    report = dict(
        rows=rows,
        controls=controls,
        pairs=16,
        joint_passes=sum(r["scores"]["joint_pass"] for r in rows),
        input_and_source_sha256=hashes,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Train16 box proxy and controls, not independent pixel accuracy.",
            "RGB input for RoMa versus grayscale XoFTR; not a weights-only ablation.",
            "Direct dense map at certainty >=0.5; no TPS or stochastic sampling.",
            "Float32 inference, default coarse560 and fine864 resolutions.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(joint_passes=report["joint_passes"], pairs=16, qualified=False)))


if __name__ == "__main__":
    main()
