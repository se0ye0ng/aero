"""Offline CPU integration probe: real tiny upstream LDM, VAE, CLIP and trainer.

All images, text assets and model weights are synthetic/random test fixtures.
This is NOT a real-data generator trainer, pretrained-model test or qualification.
Use only the isolated DiffV2IR environment; never modify the registration runtime.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aero_ir.generate.diffv2ir_data import DiffV2IRTrainDataset
from aero_ir.generate.diffv2ir_sampling import (
    official_modules,
    source_identity,
    taming_source_identity,
)
from aero_ir.utils.manifest import file_sha256


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def synthetic_dataset(root):
    """Hash actual fixture files, explicitly not a real registered export."""
    root.mkdir()
    identity = root / "fixture_identity.json"
    write_json(identity, {"synthetic_test_only": True, "qualification": "not_evaluated"})
    train = root / "train.json"
    write_json(train, {"synthetic_sequence": ["synthetic"]})
    rng = np.random.default_rng(0)
    records = []
    for i in range(4):
        assets = {}
        rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        seg = np.repeat(((np.indices((64, 64)).sum(0) % 2) * 255)[..., None], 3, 2).astype(np.uint8)
        for role, value in (
            ("rgb", rgb),
            ("ir", rgb[..., 0]),
            ("seg", seg),
            ("support", np.full((64, 64), 255, np.uint8)),
        ):
            path = root / f"{i}_{role}.png"
            Image.fromarray(value).save(path)
            assets[role] = {"path": path.name, "sha256": file_sha256(path)}
        records.append(
            {
                "pair_id": f"synthetic_{i}",
                "sequence": "synthetic_sequence",
                "visible_frame_index": i,
                "infrared_frame_index": i,
                "height": 64,
                "width": 64,
                "prompt": "synthetic test image",
                **assets,
            }
        )
    manifest = root / "conditioning.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "kind": "aero_diffv2ir_train_conditioning",
            "split": "train",
            "coordinate_domain": "common_ir_observation_grid",
            "synthetic_test_only": True,
            "official_train_sha256": file_sha256(train),
            "provenance": dict.fromkeys(
                (
                    "pair_manifest_sha256",
                    "registration_checkpoint_sha256",
                    "registration_report_sha256",
                    "export_spec_sha256",
                    "caption_spec_sha256",
                    "segmentation_spec_sha256",
                ),
                file_sha256(identity),
            ),
            "records": records,
        },
    )
    return DiffV2IRTrainDataset(
        manifest, train, file_sha256(manifest), file_sha256(train), (64, 64)
    )


def tiny_clip_assets(root):
    """Local random text encoder and tiny tokenizer; never downloaded CLIP weights."""
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    root.mkdir()
    # Each byte character can be tokenized with/without end-of-word suffix.
    from transformers.models.clip.tokenization_clip import bytes_to_unicode

    tokens = list(bytes_to_unicode().values())
    vocab = {token: i for i, token in enumerate(tokens + [t + "</w>" for t in tokens])}
    vocab["<|startoftext|>"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    write_json(root / "vocab.json", vocab)
    with (root / "merges.txt").open("x") as handle:
        handle.write("#version: 0.2\n")
    tokenizer = CLIPTokenizer(
        str(root / "vocab.json"), str(root / "merges.txt"), model_max_length=32
    )
    tokenizer.save_pretrained(root)
    config = CLIPTextConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
        bos_token_id=vocab["<|startoftext|>"],
        eos_token_id=vocab["<|endoftext|>"],
        pad_token_id=vocab["<|endoftext|>"],
    )
    CLIPTextModel(config).save_pretrained(root, safe_serialization=False)


def tiny_config(clip_root):
    return {
        "target": "aero_ir.generate.diffv2ir_training.create_training_model",
        "params": {
            "timesteps": 16,
            "linear_start": 0.00085,
            "linear_end": 0.012,
            "first_stage_key": "edited",
            "cond_stage_key": "edit",
            "image_size": 32,
            "channels": 4,
            "conditioning_key": "hybrid",
            "cond_stage_trainable": False,
            "scale_factor": 0.18215,
            "use_ema": True,
            "load_ema": True,
            "monitor": None,
            "unet_config": {
                "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {
                    "image_size": 32,
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
            },
            "first_stage_config": {
                "target": "ldm.models.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 4,
                    "lossconfig": {"target": "torch.nn.Identity"},
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 64,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 32,
                        "ch_mult": [1, 2],
                        "num_res_blocks": 1,
                        "attn_resolutions": [],
                        "dropout": 0.0,
                    },
                },
            },
            "cond_stage_config": {
                "target": "ldm.modules.encoders.modules.FrozenCLIPEmbedder",
                "params": {"version": str(clip_root.resolve()), "device": "cpu", "max_length": 32},
            },
        },
    }


def probe(upstream_root, output):
    from pytorch_lightning import Callback, Trainer

    source = source_identity(upstream_root)
    taming = taming_source_identity()
    _, omega, _, instantiate = official_modules(upstream_root)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    dataset = synthetic_dataset(output / "synthetic_input")
    tiny_clip_assets(output / "random_clip")
    config = tiny_config(output / "random_clip")
    write_json(output / "tiny_config.json", config)
    model = instantiate(omega.create(config)).cpu()
    model.learning_rate = 1e-4
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    latent, cond = model.get_input(batch, "edited", uncond=0)
    shapes = {key: list(value[0].shape) for key, value in cond.items()}
    if latent.shape != (1, 4, 32, 32) or shapes != {
        "c_crossattn": [1, 32, 32],
        "c_concat1": [1, 4, 32, 32],
        "c_concat2": [1, 4, 32, 32],
    }:
        raise ValueError("dataset/latent conditioning interface differs")
    frozen = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k.startswith(("first_stage_model.", "cond_stage_model."))
    }
    initial_denoiser = {k: v.detach().clone() for k, v in model.model.state_dict().items()}

    class Audit(Callback):
        def __init__(self):
            self.gradients, self.losses = [], []

        def on_after_backward(self, trainer, module):
            gradients = [p.grad for p in module.model.parameters() if p.grad is not None]
            valid = bool(gradients) and all(torch.isfinite(g).all() for g in gradients)
            nonzero = any(bool(g.abs().max() > 0) for g in gradients)
            if not valid or not nonzero:
                raise ValueError("missing/nonfinite/all-zero diffusion gradient")
            if any(p.grad is not None for p in module.first_stage_model.parameters()):
                raise ValueError("frozen VAE received gradient")
            if any(p.grad is not None for p in module.cond_stage_model.parameters()):
                raise ValueError("frozen text encoder received gradient")
            self.gradients.append(True)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = float(outputs["loss"].detach())
            if not np.isfinite(loss):
                raise ValueError("nonfinite diffusion loss")
            self.losses.append(loss)

    def trainer(steps, callback):
        return Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=steps,
            max_epochs=10,
            accumulate_grad_batches=2,
            precision=32,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            callbacks=[callback],
            default_root_dir=str(output),
        )

    first_audit = Audit()
    first = trainer(2, first_audit)
    first.fit(model, train_dataloaders=loader)
    if first.global_step != 2 or len(first_audit.gradients) != 4:
        raise ValueError("gradient accumulation/update budget differs")
    if all(
        torch.equal(value, model.model.state_dict()[key]) for key, value in initial_denoiser.items()
    ):
        raise ValueError("diffusion optimizer did not change weights")
    checkpoint = output / "tiny_training.ckpt"
    first.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # This file was created by this process from local synthetic inputs only.
    restored = instantiate(omega.create(config)).cpu()
    restored.learning_rate = model.learning_rate
    restored.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()):
        raise ValueError("full LDM checkpoint reload changed tensors")
    second_audit = Audit()
    second = trainer(3, second_audit)
    second.fit(restored, train_dataloaders=loader, ckpt_path=str(checkpoint))
    if second.global_step != 3 or len(second_audit.gradients) != 2:
        raise ValueError("Lightning checkpoint continuation did not add exactly one optimizer step")
    optimizer_steps = sorted({int(v["step"]) for v in second.optimizers[0].state.values()})
    if optimizer_steps != [3]:
        raise ValueError("Adam state did not continue from the saved optimizer")
    if not all(torch.equal(v, restored.state_dict()[k]) for k, v in frozen.items()):
        raise ValueError("frozen encoder weights changed")
    if source_identity(upstream_root) != source or taming_source_identity() != taming:
        raise ValueError("upstream source changed during probe")
    return {
        "upstream": source,
        "taming_upstream": taming,
        "conditioning_shapes": shapes,
        "latent_shape": list(latent.shape),
        "initial_optimizer_steps": first.global_step,
        "resumed_optimizer_steps": second.global_step,
        "adam_steps_after_resume": optimizer_steps,
        "checkpoint_tensor_reload_equal": True,
        "finite_nonzero_gradients": True,
        "frozen_encoder_weights_unchanged": True,
        "first_microbatch_losses": first_audit.losses,
        "resumed_microbatch_losses": second_audit.losses,
        "ema_updates_after_resume": int(restored.model_ema.num_updates),
        "ema_update_scope": "upstream on_train_batch_end, not optimizer-step frequency",
        "checkpoint_sha256": file_sha256(checkpoint),
        "synthetic_manifest_sha256": file_sha256(output / "synthetic_input/conditioning.json"),
        "random_clip_assets_sha256": {
            p.name: file_sha256(p) for p in sorted((output / "random_clip").iterdir())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=Path("experiments/external/DiffV2IR"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("refusing to overwrite runtime evidence; use a fresh output")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    names = (
        "scripts/check_diffv2ir_training_runtime.py",
        "src/aero_ir/generate/diffv2ir_data.py",
        "src/aero_ir/generate/diffv2ir_training.py",
        "src/aero_ir/generate/diffv2ir_sampling.py",
        "requirements/diffv2ir-runtime.txt",
    )
    sources = {name: file_sha256(project / name) for name in names}
    result = {
        "kind": "synthetic_tiny_diffv2ir_full_training_cpu_probe",
        "source_sha256": sources,
        "torch_version": str(torch.__version__),
        "real_dataset_used": False,
        "pretrained_weights_used": False,
        "gpu_tested": False,
        "generator_training_eligible": "hold_not_qualified",
        "resume_is_bitwise_reproducibility_proof": False,
    }
    try:
        result.update(probe(args.upstream_root, args.out_dir))
        if any(file_sha256(project / name) != sha for name, sha in sources.items()):
            raise ValueError("probe source drift")
        result["ok"] = True
    except Exception as error:
        result["ok"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(args.out_dir / "report.json", result)
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "report": str(args.out_dir / "report.json"),
                "error": result.get("error"),
            }
        )
    )
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
