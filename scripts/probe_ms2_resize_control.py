"""Post-screen same-image resize control; not cross-modal registration evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from aero_ir.registration.detector_free import infer, load_matcher
from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_registration_landmarks import resize_gray, to_native
from scripts.probe_registration_rgb_resolution import ignored_bytecode_paths


def resize_correspondence(points, source_wh, target_wh):
    """OpenCV pixel-centre coordinates for a known resize, not fitted matches."""
    points = np.asarray(points, dtype=np.float64)
    source_wh, target_wh = np.asarray(source_wh), np.asarray(target_wh)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or source_wh.shape != (2,) or target_wh.shape != (2,)
            or not np.isfinite(source_wh).all() or not np.isfinite(target_wh).all()
            or (source_wh <= 0).any() or (target_wh <= 0).any()):
        raise ValueError("finite Nx2 points and positive width/height pairs required")
    return (points + .5) * (target_wh / source_wh) - .5


def residual_summary(delta):
    delta = np.asarray(delta)
    errors = np.linalg.norm(delta, axis=1)
    return dict(matches=len(delta),
                median_target_px=float(np.median(errors)) if len(delta) else None,
                p95_target_px=float(np.percentile(errors, 95)) if len(delta) else None,
                median_signed_xy=np.median(delta, axis=0).tolist() if len(delta) else None,
                within_target_px_counts={str(t): int(np.count_nonzero(errors <= t))
                                         for t in (1., 3., 5., 10.)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path("configs/experiment/registration_ms2_image_cpu.yaml")
    config = yaml.safe_load(config_path.read_text())
    source_report = Path("experiments/ms2_image_matching_01/report.json")
    expected = "3abbeca131b191c85c24357ea6758c0acae9dcbdd51fffd5a066e3718d717d39"
    if file_sha256(source_report) != expected:
        raise ValueError("original image screen changed")
    report = json.loads(source_report.read_text())
    hashes = report["input_and_source_sha256"].copy()
    hashes[str(source_report)] = expected
    hashes[str(Path(__file__))] = file_sha256(__file__)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed input/source: {path}")
    plan = json.loads(Path(config["plan"]).read_text())
    vendor = Path(config["vendor"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != config["vendor_commit"]:
        raise ValueError("changed vendor revision")
    ignored = ignored_bytecode_paths(subprocess.check_output(
        ["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=all"], text=True))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="aero-ms2-resize-bytecode-")
    sys.dont_write_bytecode = True
    args.out_dir.mkdir(parents=True, exist_ok=False)
    preflight = dict(input_and_source_sha256=hashes, frames=plan["frame_ids"],
                     source_report_sha256=expected, target_wh=[640, 256],
                     protocol="post-screen diagnostic; same RGB image, known anisotropic resize",
                     threshold_role="diagnostic counts only; no qualification gate")
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    rows = []
    with torch.inference_mode():
        for name, spec in config["models"].items():
            model = load_matcher("xoftr", Path(spec["path"]), vendor, "cpu")
            for frame in plan["frame_ids"]:
                path = (Path(config["sync_root"]) / "sync_data" / plan["sequence"]
                        / "rgb/img_left" / f"{frame}.png")
                with Image.open(path) as image:
                    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                shaped = cv2.resize(gray, (640, 256), interpolation=cv2.INTER_AREA)
                inputs = [resize_gray(g, config["long_side"]) for g in (gray, shaped)]
                native_wh = [gray.shape[::-1], shaped.shape[::-1]]
                for source, target in ((0, 1), (1, 0)):
                    a, b, confidence, flags = infer(model, "xoftr", inputs[source][0],
                                                    inputs[target][0], "cpu")
                    a, b = to_native(a, inputs[source][1]), to_native(b, inputs[target][1])
                    expected_points = resize_correspondence(a, native_wh[source], native_wh[target])
                    delta = b - expected_points
                    artifact = args.out_dir / f"{name}_{frame}_{source}.npz"
                    np.savez_compressed(artifact, source_xy=a, target_xy=b,
                                        expected_target_xy=expected_points, confidence=confidence)
                    row = dict(model=name, frame=frame, source=source, target=target,
                               source_wh=native_wh[source], target_wh=native_wh[target],
                               flags=flags, artifact=artifact.name, sha256=file_sha256(artifact),
                               **residual_summary(delta))
                    rows.append(row)
                print(f"{name} {frame}: both known-resize directions complete", flush=True)
            del model
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input/source changed during inference: {path}")
    result = dict(schema="ms2_known_resize_control_v1", device="cpu", rows=rows,
                  input_and_source_sha256=hashes, vendor_bytecode_ignored=ignored,
                  preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
                  registration_qualified=False, generator_training_approved=False,
                  note="Same-modality synthetic-coordinate control, not physical RGB-IR evidence")
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
