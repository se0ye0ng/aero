"""Reverse image matching to audit the reference observations, not certify pixel GT."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_ir.registration.calibrated import project_pixels
from aero_ir.registration.detector_free import infer, load_matcher, reciprocal_mask
from aero_ir.registration.ms2_image_reference import display_thermal
from aero_ir.utils.manifest import file_sha256
from scripts.probe_external_registration_landmarks import resize_gray, to_native
from scripts.probe_ms2_confirmation import add_identity, read
from scripts.probe_ms2_intrinsic_confirmation import load_protocol
from scripts.probe_ms2_stereo_depth import summarize
from scripts.probe_registration_rgb_resolution import ignored_bytecode_paths

SOURCE = Path("experiments/ms2_intrinsic_confirmation_image_01/report.json")
SHA = "e2b4e5ea6695204ee28a23349611de05ca4eecbbca5b815f46115136d4995d04"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha256(SOURCE) != SHA:
        raise ValueError("forward evidence changed")
    source = json.loads(SOURCE.read_text())
    plan, config, hashes, files, calib, _, transforms, _ = load_protocol()
    for p, digest in source["input_and_source_sha256"].items():
        if p in hashes and hashes[p] != digest:
            raise ValueError("conflicting source identity")
        hashes[p] = digest
    for row in source["rows"]:
        p = SOURCE.parent / row["artifact"]
        if p.resolve().parent != SOURCE.parent.resolve() or file_sha256(p) != row["sha256"]:
            raise ValueError("changed or unsafe forward artifact")
        add_identity(hashes, p)
    for p in (SOURCE, Path(__file__), Path("src/aero_ir/registration/detector_free.py")):
        add_identity(hashes, p)
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed source: {p}")
    sys.dont_write_bytecode = True
    vendor = Path(config["vendor"]).resolve()
    if (
        subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
        != config["vendor_commit"]
    ):
        raise ValueError("vendor revision changed")
    ignored = ignored_bytecode_paths(
        subprocess.check_output(
            ["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=all"], text=True
        )
    )
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    prepared = {}
    for frame in plan["frame_ids"]:
        rgb = read(files[f"sync_data/{plan['sequence']}/rgb/img_left/{frame}.png"])
        thr = read(files[f"sync_data/{plan['sequence']}/thr/img_left/{frame}.png"])
        prepared[frame] = dict(
            rgb=resize_gray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 640),
            thr=resize_gray(display_thermal(thr, [3308.0, 4974.0]), 640),
        )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    pre = dict(
        input_and_source_sha256=hashes,
        tolerances_native_pixels=[1.0, 3.0],
        tolerance_scope="each endpoint in its own native pixel coordinates",
        device="cpu",
        torch_version=torch.__version__,
        opencv_version=cv2.__version__,
        cameras_refitted=False,
        all_forward_observations_retained=True,
        interpretation="reciprocity is consistency, NOT physical accuracy",
        vendor_ignored_bytecode=ignored,
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(pre, indent=2, allow_nan=False))
    rows = []
    with torch.inference_mode():
        for name, spec in config["models"].items():
            if file_sha256(spec["path"]) != spec["sha256"]:
                raise ValueError("checkpoint changed")
            model = load_matcher("xoftr", Path(spec["path"]), vendor, "cpu")
            for row in [r for r in source["rows"] if r["model"] == name]:
                a = prepared[row["target_frame"]]["thr"]
                b = prepared[row["frame"]]["rgb"]
                p0, p1, confidence, flags = infer(model, "xoftr", a[0], b[0], "cpu")
                p0, p1 = to_native(p0, a[1]), to_native(p1, b[1])
                with np.load(SOURCE.parent / row["artifact"], allow_pickle=False) as f:
                    original = dict(f)
                masks = {
                    str(t): reciprocal_mask(original["source_xy"], original["target_xy"], p0, p1, t)
                    for t in (1.0, 3.0)
                }
                path = args.out_dir / row["artifact"]
                np.savez_compressed(
                    path,
                    reverse_thermal_xy=p0,
                    reverse_rgb_xy=p1,
                    reverse_confidence=confidence,
                    **{f"reciprocal_{k}": v for k, v in masks.items()},
                )
                cameras = {}
                for camera, candidate in plan["candidates"].items():
                    projected = project_pixels(
                        original["source_xy"],
                        original["stereo_depth_m"],
                        calib["K_rgbL"],
                        np.asarray(candidate["target_intrinsic"]),
                        np.asarray(candidate["transform"]) @ transforms[row["frame"]],
                        source_shape=(384, 1224),
                        target_shape=(256, 640),
                    )
                    errors = np.linalg.norm(projected.target_xy - original["target_xy"], axis=1)
                    keep = original["author_supported"]
                    errors[~keep | ~projected.supported] = np.inf
                    scores = summarize(errors[keep])
                    if scores != row["scores"][camera]["fixed_supported"]:
                        raise ValueError("forward reference scores differ")
                    cameras[camera] = dict(
                        all_author_supported=scores,
                        reciprocal={
                            k: dict(
                                conditional=summarize(errors[keep & mask]),
                                fraction_original_reference_reciprocal_and_within_3px=float(
                                    np.count_nonzero(keep & mask & (errors <= 3)) / keep.sum()
                                )
                                if keep.any()
                                else None,
                            )
                            for k, mask in masks.items()
                        },
                    )
                rows.append(
                    {k: row[k] for k in ("model", "frame", "target_frame", "condition")}
                    | dict(
                        forward_matches=len(original["source_xy"]),
                        reverse_matches=len(p0),
                        reciprocal_counts={k: int(v.sum()) for k, v in masks.items()},
                        flags=flags,
                        cameras=cameras,
                        artifact=path.name,
                        sha256=file_sha256(path),
                    )
                )
                print(name, row["frame"], row["condition"], "reverse matching complete", flush=True)
            del model
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"input changed during run: {p}")
    result = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        preflight_sha256=file_sha256(args.out_dir / "preflight.json"),
        registration_qualified=False,
        generator_training_approved=False,
        note="No frame excluded, no camera fitted, no pseudo-GT generated. "
        "Conditional subsets diagnose observation consistency only.",
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}; not qualification")


if __name__ == "__main__":
    main()
