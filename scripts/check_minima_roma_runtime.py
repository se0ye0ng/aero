"""Offline CPU constructor check; not neural inference or registration qualification."""

import argparse
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from aero_ir.utils.manifest import file_sha256
from scripts.fetch_minima_roma import COMMIT, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    source = ROOT / f"RoMa_minima-{COMMIT}"
    manifest = ROOT / "download_manifest.json"
    assets = json.loads(manifest.read_text())
    hashes = {str(manifest): file_sha256(manifest), str(Path(__file__)): file_sha256(__file__)}
    for name, entry in assets.items():
        path = ROOT / name
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"asset changed: {path}")
        hashes[str(path)] = entry["sha256"]
    for path in source.rglob("*.py"):
        hashes[str(path)] = file_sha256(path)
    hashes[str(source / "LICENSE")] = file_sha256(source / "LICENSE")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source.resolve()))
    error = RuntimeError("implicit network/download forbidden during offline preflight")
    with (
        patch.object(torch.hub, "load_state_dict_from_url", side_effect=error),
        patch.object(torch.hub, "load", side_effect=error),
        patch.object(socket.socket, "connect", side_effect=error),
    ):
        from romatch import roma_outdoor

        weights = torch.load(ROOT / "minima_roma.pth", map_location="cpu", weights_only=True)
        backbone = torch.load(
            ROOT / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True
        )
        print("Verified weights loaded; constructing on CPU...", flush=True)
        model = roma_outdoor(device="cpu", weights=weights, dinov2_weights=backbone)
        model.eval()
    params = sum(p.numel() for p in model.parameters())
    dino = model.encoder.dinov2_vitl14[0]
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"source changed: {path}")
    report = dict(
        ok=True,
        device="cpu",
        torch_version=torch.__version__,
        constructor_parameters=params,
        unregistered_backbone_parameters=sum(p.numel() for p in dino.parameters()),
        neural_inference_run=False,
        registration_qualified=False,
        generator_training_approved=False,
        input_and_source_sha256=hashes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "input_and_source_sha256"}))


if __name__ == "__main__":
    main()
