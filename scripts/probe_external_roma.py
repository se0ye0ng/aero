"""Frozen four-pair external landmark diagnostic for MINIMA-RoMa, without GT fitting."""

import argparse
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from aero_ir.registration.roma_dense import dense_predictor, split_dense
from aero_ir.utils.manifest import file_sha256
from scripts.fetch_minima_roma import COMMIT, ROOT
from scripts.probe_external_registration_landmarks import summarize, validate_landmarks
from scripts.probe_external_resolution import protocol as external_protocol
from scripts.probe_minima_roma_gpu import protocol as roma_protocol
from scripts.verify_antiuav_tiled_matching import safe_artifact


def protocol():
    config, _, manifest, external = external_protocol()
    roma, _, _ = roma_protocol()
    hashes = external.copy()
    for key, value in roma.items():
        if key in hashes and hashes[key] != value:
            raise ValueError("conflicting source identity")
        hashes[key] = value
    for path in (Path(__file__), Path("scripts/run_external_roma_gpu.sh")):
        hashes[str(path)] = file_sha256(path)
    return Path(config["cache"]), manifest["pairs"], hashes


def score(warp, certainty, sizes, references):
    maps = split_dense(warp, certainty)
    result = {}
    for name, gated in (("fixed_confidence_ge_half", True), ("ungated_diagnostic", False)):
        directions = []
        for d, (field, confidence) in enumerate(maps):
            predictor = dense_predictor(
                field, confidence if gated else np.ones_like(confidence), sizes[d], sizes[1 - d]
            )
            error = np.linalg.norm(predictor(references[d]) - references[1 - d], axis=1)
            error[~np.isfinite(error)] = np.inf
            directions.append(
                dict(
                    target="thermal" if d == 0 else "rgb",
                    units="native_target_file_pixels",
                    errors=[float(e) if np.isfinite(e) else None for e in error],
                    summary=summarize(error, [1.0, 3.0, 5.0, 10.0]),
                )
            )
        result[name] = directions
    return result


def inputs(root, pair):
    images, references = [], []
    for image_name, point_name in (("V.JPG", "points_rgb.txt"), ("T.JPG", "points_thermal.txt")):
        with Image.open(root / pair / image_name) as image:
            images.append(image.convert("RGB"))
        references.append(
            validate_landmarks(np.loadtxt(root / pair / point_name), images[-1].size[::-1])
        )
    if references[0].shape != references[1].shape:
        raise ValueError("unpaired references")
    return images, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "verify"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root, pairs, hashes = protocol()
    data = [inputs(root, pair) for pair in pairs]
    if args.action == "verify":
        report = json.loads((args.out_dir / "report.json").read_text())
        if (
            report["input_and_source_sha256"] != hashes
            or report["registration_qualified"]
            or report["generator_training_approved"]
        ):
            raise ValueError("provenance or approval differs")
        if [r["pair"] for r in report["rows"]] != pairs:
            raise ValueError("pair inventory differs")
        for row, (images, references) in zip(report["rows"], data, strict=True):
            with np.load(safe_artifact(args.out_dir, row), allow_pickle=False) as saved:
                scores = score(
                    saved["warp"], saved["certainty"], [im.size for im in images], references
                )
            if scores != row["scores"]:
                raise ValueError("saved score differs")
        print(
            json.dumps(
                dict(
                    ok=True,
                    pairs=len(pairs),
                    neural_inference_replayed=False,
                    registration_qualified=False,
                )
            )
        )
        return
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if args.action == "preflight":
        print(
            json.dumps(
                dict(
                    ok=True,
                    pairs=pairs,
                    landmarks=sum(len(r[0]) for _, r in data),
                    input_identities=len(hashes),
                    gpu_run=False,
                )
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((ROOT / f"RoMa_minima-{COMMIT}").resolve()))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    error = RuntimeError("implicit network/download forbidden")
    with (
        patch.object(torch.hub, "load_state_dict_from_url", side_effect=error),
        patch.object(torch.hub, "load", side_effect=error),
        patch.object(socket.socket, "connect", side_effect=error),
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
        for i, (pair, (images, references)) in enumerate(zip(pairs, data, strict=True)):
            # No landmark coordinates are passed to the network or used for fitting.
            warp, certainty = model.match(*images, device="cuda")
            w, c = warp.float().cpu().numpy(), certainty.float().cpu().numpy()
            scores = score(w, c, [im.size for im in images], references)
            artifact = args.out_dir / f"{i:03d}_dense.npz"
            np.savez_compressed(artifact, warp=w, certainty=c)
            rows.append(
                dict(pair=pair, scores=scores, artifact=artifact.name, sha256=file_sha256(artifact))
            )
            print(json.dumps(dict(pair=pair, scores=scores)), flush=True)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    report = dict(
        rows=rows,
        input_and_source_sha256=hashes,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Four development pairs, not held-out qualification.",
            "Ungated branch is diagnostic, not a replacement acceptance gate.",
            "External landmark accuracy does not qualify Anti-UAV300.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
