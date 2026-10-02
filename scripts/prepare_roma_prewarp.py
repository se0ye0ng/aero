"""Prepare image-only trained coarse warps for a separate residual matcher probe."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.prewarp import field_predictor
from aero_ir.utils.manifest import file_sha256
from scripts.audit_antiuav300_dense_registration import _read_at
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_minima_roma_gpu import protocol
from scripts.train_registration_replay import load_checkpoint

CHECKPOINT = Path("experiments/registration_replay_e300_seed0/uniform/final.pth")
SPEC = CHECKPOINT.parent / "run_spec.json"
PINS = {
    CHECKPOINT: "f7a057317f8c8348760210c7750230b4ed37f5e689e102426fdad3911b5b647e",
    SPEC: "a0997a24f3974a31b7be48e80adabf3f10d92a95d10c9980eb72cb304e4d8ac1",
}


def render(visible, infrared_size, ir_to_rgb, rgb_top):
    w, h = infrared_size
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    mapped = ir_to_rgb(np.c_[x.ravel(), y.ravel()]).reshape(h, w, 2)
    support = np.isfinite(mapped).all(2) & (mapped[..., 1] >= rgb_top)
    safe = np.where(support[..., None], mapped, -1).astype(np.float32)
    rendered = cv2.remap(
        visible, safe[..., 0], safe[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return rendered, support


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300"))
    args = p.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    hashes, cases, metadata = protocol()
    spec = json.loads(SPEC.read_text())
    for path, sha in {**spec["source_sha256"], **PINS}.items():
        if file_sha256(path) != sha:
            raise ValueError(f"coarse model source/input changed: {path}")
        hashes[str(path)] = sha
    for path in (Path(__file__), Path("src/aero_ir/registration/prewarp.py")):
        hashes[str(path)] = file_sha256(path)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model, state = load_checkpoint(CHECKPOINT, spec, torch.device("cpu"))
    if state["completed_epochs"] != 300 or spec["arm"] != "uniform":
        raise ValueError("completed uniform300 initializer required")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.inference_mode():
        for i, case in enumerate(cases):
            native = []
            for name in ("visible", "infrared"):
                path = args.root / "train" / case["sequence_id"] / f"{name}.mp4"
                cap = cv2.VideoCapture(str(path))
                try:
                    frame = _read_at(cap, case["frame_index"], path)
                finally:
                    cap.release()
                expected = metadata[case["sequence_id"]][name]["decoded_native_sha256"]
                if hashlib.sha256(frame.tobytes()).hexdigest() != expected:
                    raise ValueError("native frame differs from fixed panel")
                native.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensors = [
                torch.from_numpy(cv2.resize(im, (256, 256))).permute(2, 0, 1)[None].float() / 255
                for im in native
            ]
            first, second = [f[0].cpu().numpy() for f in model.fields(tensors[1], tensors[0])]
            # first is IR -> RGB, second is RGB -> IR; full-image centre domains.
            sizes = [
                tuple(metadata[case["sequence_id"]][name]["resized_shape"][::-1])
                for name in ("visible", "infrared")
            ]
            rgb_to_ir = field_predictor(second, sizes[0], sizes[1])
            ir_to_rgb = field_predictor(first, sizes[1], sizes[0])
            native_rgb_size = native[0].shape[1::-1]
            native_map = field_predictor(first, sizes[1], native_rgb_size)
            rgb_top, ir_top = case["header_rows"]
            rendered, support = render(
                native[0], sizes[1], native_map, rgb_top / sizes[0][1] * native_rgb_size[1]
            )
            infrared = cv2.resize(native[1], sizes[1])
            support[:ir_top] = False
            # RGB HUD also excluded at its mapped native location above.
            rendered[~support] = 0
            path = args.out_dir / f"{i:03d}_prewarp.npz"
            np.savez_compressed(
                path,
                warped_rgb=rendered[ir_top:],
                infrared=infrared[ir_top:],
                observed=support,
                ir_to_rgb_field=first,
                rgb_to_ir_field=second,
            )
            ious = [
                corner_iou(fn, case["boxes_xyxy"][d], case["boxes_xyxy"][1 - d])
                for d, fn in enumerate((rgb_to_ir, ir_to_rgb))
            ]
            rows.append(
                dict(
                    sequence_id=case["sequence_id"],
                    frame_index=case["frame_index"],
                    image_sizes=sizes,
                    header_rows=case["header_rows"],
                    boxes_xyxy=case["boxes_xyxy"],
                    coarse_box_iou=ious,
                    observed_crop_fraction=float(support[ir_top:].mean()),
                    artifact=path.name,
                    sha256=file_sha256(path),
                )
            )
            print(f"{i + 1}/16: coarse IoU={ious}", flush=True)
    for path, sha in hashes.items():
        if file_sha256(path) != sha:
            raise ValueError(f"input changed during preparation: {path}")
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        coarse_box_proxy_both_ge_06=sum(
            all(v is not None and v >= 0.6 for v in r["coarse_box_iou"]) for r in rows
        ),
        registration_qualified=False,
        generator_training_approved=False,
        inference_device="cpu",
        refinement_inference_completed=False,
        limitations=[
            "Coarse checkpoint trained on these sequences; no held-out claim.",
            "Boxes enter scoring only; image warp has no GT input.",
            "Bilinear rendering and missing support do not create new evidence.",
        ],
    )
    with (args.out_dir / "manifest.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(coarse_box_proxy_both_ge_06=result["coarse_box_proxy_both_ge_06"])))


if __name__ == "__main__":
    main()
