"""Offline CPU compatibility probe for actual upstream modules; no weight download.

A small randomly initialized upstream UNet takes one optimizer step. This is a
component compatibility test, NOT trained IR generation or a model qualification.
Run only with the separately installed DiffV2IR runtime and PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from aero_ir.generate.diffv2ir_sampling import (
    official_modules,
    source_identity,
    taming_source_identity,
)
from aero_ir.utils.manifest import file_sha256


def component_probe(root):
    source = source_identity(root)
    taming_source = taming_source_identity()
    _, _, upstream, instantiate = official_modules(root)
    imports = {}
    for name in (
        "ldm.models.diffusion.ddpm_DiffV2IR",
        "ldm.models.autoencoder",
        "ldm.modules.encoders.modules",
        "ldm.modules.diffusionmodules.openaimodel",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(root.resolve() / "stable_diffusion"):
            raise ValueError(f"upstream module shadowed by another installation: {name}")
        imports[name] = {"path": str(path), "sha256": file_sha256(path)}
    torch.manual_seed(0)
    config = {
        "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
        "params": {
            "image_size": 8,
            "in_channels": 12,
            "out_channels": 4,
            "model_channels": 32,
            "num_res_blocks": 1,
            "attention_resolutions": [1],
            "channel_mult": [1, 2],
            "num_heads": 4,
            "use_spatial_transformer": True,
            "transformer_depth": 1,
            "context_dim": 32,
            "use_checkpoint": False,
            "legacy": False,
        },
    }
    model = instantiate(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image = torch.randn(1, 12, 8, 8)
    context = torch.randn(1, 3, 32)
    timesteps = torch.tensor([1])
    before = model(image, timesteps, context=context)
    loss = (before - 1).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("missing/nonfinite upstream UNet gradients")
    if not any(bool(g.abs().max() > 0) for g in gradients):
        raise ValueError("all upstream UNet gradients are zero")
    optimizer.step()
    with torch.no_grad():
        after = model(image, timesteps, context=context)
    if not torch.isfinite(after).all() or torch.equal(before, after):
        raise ValueError("upstream UNet optimizer step produced no finite change")

    class AnalyticDenoiser(torch.nn.Module):
        """Known arithmetic only, to test upstream CFG branch ordering."""

        def forward(self, z, sigma, cond):
            text = cond["c_crossattn"][0].mean((1, 2))[:, None, None, None]
            return torch.zeros_like(z) + text + cond["c_concat1"][0] + cond["c_concat2"][0]

    guided = upstream.CFGDenoiser(AnalyticDenoiser())
    z = torch.zeros(1, 4, 2, 2)
    cond = {
        "c_crossattn": [torch.full((1, 2, 3), 5.0)],
        "c_concat1": [torch.full_like(z, 3)],
        "c_concat2": [torch.full_like(z, 2)],
    }
    uncond = {k: [torch.zeros_like(v[0])] for k, v in cond.items()}
    cfg_checks = []
    for ts, ims, ss in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (7.5, 1.5, 1.5)):
        predicted = guided(z, torch.ones(1), cond, uncond, ts, ims, ss)
        expected = 5 * ts + 3 * ims + 2 * ss
        if not torch.allclose(predicted, torch.full_like(z, expected)):
            raise ValueError("official guidance branch semantics differ from the bridge")
        cfg_checks.append({"scales": [ts, ims, ss], "expected": expected, "ok": True})
    if source_identity(root) != source or taming_source_identity() != taming_source:
        raise ValueError("upstream source changed during runtime probe")
    return {
        "upstream": source,
        "taming_upstream": taming_source,
        "imported_modules": imports,
        "random_small_unet": {
            "config": config,
            "loss": float(loss.detach()),
            "output_shape": list(after.shape),
            "optimizer_steps": 1,
            "finite_nonzero_gradients": True,
            "output_changed": True,
        },
        "guidance_analytic_fixture_checks": cfg_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=Path("experiments/external/DiffV2IR"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite runtime evidence; use a fresh output")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    project = Path(__file__).resolve().parents[1]
    code_hashes = {
        name: file_sha256(project / name)
        for name in (
            "scripts/check_diffv2ir_runtime.py",
            "src/aero_ir/generate/diffv2ir_sampling.py",
            "requirements/diffv2ir-runtime.txt",
        )
    }
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
    )
    report = {
        "kind": "diffv2ir_cpu_runtime_component_probe",
        "probe_source_sha256": code_hashes,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "built_cuda": torch.version.cuda,
        "pip_check": {
            "returncode": pip_check.returncode,
            "stdout": pip_check.stdout,
            "stderr": pip_check.stderr,
        },
        "installed_distributions": sorted(
            [d.metadata["Name"], d.version]
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        ),
        "full_pretrained_model_tested": False,
        "gpu_tested": False,
        "generator_training_eligible": "hold_not_qualified",
    }
    try:
        if pip_check.returncode:
            raise ValueError("runtime pip check failed")
        report.update(component_probe(args.upstream_root))
        if any(file_sha256(project / name) != digest for name, digest in code_hashes.items()):
            raise ValueError("probe sources changed during execution")
        report["ok"] = True
    except Exception as error:
        report["ok"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["cause"] = str(error.__cause__) if error.__cause__ else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"ok": report["ok"], "report": str(args.out), "error": report.get("error")}))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
