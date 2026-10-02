"""Actual CPU inference and native-image diagnostics for completed residual pilots.

First two sorted sequence IDs per paired pass/fail stratum, including regressions.
Neither boxes nor these selected pictures are independent pixel correspondence GT.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.geometry import warp
from aero_ir.registration.qualification_v4 import direction_statistics
from aero_ir.utils.manifest import file_sha256
from scripts.audit_antiuav300_dense_registration import _candidate_pairs, _read_at
from scripts.audit_registration_residual_pilot import audit_sources, pilot_hashes
from scripts.compare_registration_residual_pilot import compare
from scripts.render_registration_v7 import crop_limits, plt, render, selected_cases
from scripts.train_antiuav300_registration_v7 import read_batch, tensors
from scripts.train_registration_residual_pilot import ARMS, load_checkpoint


def native_images(root, cache, sid, sample, arrays):
    """Recover exact midpoint native frames; verify resized bytes against training."""
    folder = (root / "train" / sid).resolve()
    if folder.parent != (root / "train").resolve():
        raise ValueError("sequence outside train")
    position = sample["selection"][0][1]
    indices, source, target = _candidate_pairs(folder)
    shard = json.loads((cache / "shards" / sid / "manifest.json").read_text())
    if len(indices) != shard["pairs"] or position != len(indices) // 2:
        raise ValueError("native midpoint and cached sequence differ")
    frame = indices[position]
    images, metadata = [], {}
    for name, box, box_key in zip(("visible", "infrared"), (source[position], target[position]),
                                  ("source_boxes", "target_boxes"), strict=True):
        ann_hash = file_sha256(folder / f"{name}.json")
        if ann_hash != shard[f"{name}_annotations_sha256"]:
            raise ValueError("native annotation differs from training cache")
        capture = cv2.VideoCapture(str(folder / f"{name}.mp4"))
        try:
            bgr = _read_at(capture, frame, folder / f"{name}.mp4")
        finally:
            capture.release()
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = arrays[name].shape[1:3]
        if (not np.array_equal(cv2.resize(image, (w, h)), arrays[name][0])
                or not np.array_equal(np.asarray(box, np.float32), arrays[box_key][0])):
            raise ValueError("decoded native frame/box does not reproduce cached observation")
        if file_sha256(folder / f"{name}.json") != ann_hash:
            raise ValueError("annotation changed while decoding")
        images.append(image)
        metadata[name] = {"annotation_sha256": ann_hash, "native_shape": list(image.shape),
                          "decoded_native_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    return images, {"native_frame_index": frame, "metadata": metadata}


def render_native(path, images, boxes, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, image, box, name in zip(axes, images, boxes, ("Native RGB", "Native IR"), strict=True):
        h, w = image.shape[:2]
        ax.imshow(image, extent=(0, w, h, 0), interpolation="nearest")
        x0, x1, y1, y0 = crop_limits(box, image.shape)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(f"{name}: own annotated target window")
        ax.axis("off")
    fig.suptitle(title)
    fig.text(.5, .02, "Unwarped native observations; independent crop windows, NOT registered GT.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    p.add_argument("--run-root", type=Path,
                   default=Path("experiments/registration_residual_pilot_e10_seed0"))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        p.error("refusing to overwrite diagnostic")
    identities = pilot_hashes(args.run_root)
    proof = compare(args.run_root, args.cache_root)
    if pilot_hashes(args.run_root) != identities:
        raise ValueError("pilot changed during verification")
    sources = audit_sources()
    for name in ("scripts/render_registration_residual.py", "scripts/render_registration_v7.py"):
        sources[name] = file_sha256(name)
    screens = [json.loads((args.run_root / arm / "final_train_screen.json").read_text())
               for arm in ARMS]
    groups, selected = selected_cases(*(s["rows"] for s in screens))
    samples = {r["sequence_id"]: s for r, s in zip(screens[0]["rows"], screens[0]["samples"],
                                                strict=True)}
    torch.set_num_threads(1)
    device = torch.device("cpu")
    models = {a: load_checkpoint(args.run_root / a / "final.pth", device)[0] for a in ARMS}
    args.out_dir.mkdir(parents=True)
    outputs = []
    with torch.no_grad():
        for i, (group, sid) in enumerate(selected):
            sample = samples[sid]
            arrays, digest = read_batch(args.cache_root, sample["selection"])
            if digest != sample["array_sha256"]:
                raise ValueError("cached input differs from saved screen")
            vis, ir, vb, ib = tensors(arrays, device)
            images = [arrays["visible"][0], arrays["infrared"][0]]
            measurements = {}
            for arm, model in models.items():
                f, r = model.fields(ir, vis)
                rgb = warp(vis, f)[0].permute(1, 2, 0).clamp(0, 1).numpy()
                images.append(np.rint(rgb * 255).astype(np.uint8))
                measurements[arm] = {
                    "ir_to_rgb_points": direction_statistics(f, r, ib, vb)[0],
                    "rgb_to_ir_points": direction_statistics(r, f, vb, ib)[0]}
            name = f"{i:02d}_{group}"
            path = args.out_dir / f"{name}.png"
            render(path, images, arrays["source_boxes"][0], arrays["target_boxes"][0],
                   f"{sid} / {group} (relative to continuation)\n"
                   "Train diagnostic, not qualification",
                   titles=("Visible input", "IR observation", "Continuation RGB warp",
                           "Residual RGB warp"))
            native, metadata = native_images(args.root, args.cache_root, sid, sample, arrays)
            native_path = args.out_dir / f"{name}_native.png"
            render_native(native_path, native,
                          [arrays[k][0] for k in ("source_boxes", "target_boxes")], sid)
            outputs.append({"sequence_id": sid, "group": group, "sample": sample,
                            **metadata, "cpu_metrics": measurements,
                            "images_sha256": {path.name: file_sha256(path),
                                              native_path.name: file_sha256(native_path)}})
            print(f"rendered {sid} {group}", flush=True)
    if pilot_hashes(args.run_root) != identities:
        raise ValueError("pilot drift during rendering")
    if any(file_sha256(n) != h for n, h in sources.items()):
        raise ValueError("source drift during rendering")
    report = {"kind": "residual_pilot_stratified_train_visuals_not_qualification",
              "source_sha256": sources, "pilot_artifacts_sha256": identities,
              "stratum_populations": {k: len(v) for k, v in groups.items()},
              "selection": "first two sorted IDs per paired final-pass stratum",
              "comparison_matched_budget_and_data": proof["matched_budget_and_data"],
              "outputs": outputs, "device": "cpu", "validation_or_test_access": "none",
              "generator_training_eligible": "hold_not_qualified",
              "limitations": ["Stratified train pictures, not representative performance.",
                              "Boxes/crops are not corresponding physical surface labels.",
                              "Fresh CPU inference, not bitwise GPU replay.",
                              "Native decode is verified against cached256px inputs."]}
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
