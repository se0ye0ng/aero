"""CPU fixture/config tests only; the real upstream runtime probe is separate."""

import json

import pytest

from aero_ir.generate import diffv2ir_training
from aero_ir.generate.diffv2ir_training import Lightning19BatchHook
from scripts.check_diffv2ir_training_runtime import synthetic_dataset, tiny_config


def test_synthetic_fixture_is_explicit_and_not_qualification(tmp_path):
    dataset = synthetic_dataset(tmp_path / "synthetic")
    assert len(dataset) == 4
    manifest = json.loads(dataset.manifest_path.read_text())
    assert manifest["synthetic_test_only"] is True
    for i in range(4):
        item = dataset[i]
        assert item["edited"].shape == (3, 64, 64)
        assert item["edit"]["c_concat1"].shape == (3, 64, 64)
        assert item["edit"]["c_concat2"].shape == (3, 64, 64)
        assert dataset.sample_provenance(i)["registration_qualification"] == (
            "not_established_by_data_loader"
        )


def test_tiny_config_uses_real_upstream_components_with_local_cpu_conditioner(tmp_path):
    config = tiny_config(tmp_path)
    assert config["target"] == "aero_ir.generate.diffv2ir_training.create_training_model"
    params = config["params"]
    assert "ckpt_path" not in params
    assert params["cond_stage_config"]["params"]["version"] == str(tmp_path.resolve())
    assert params["cond_stage_config"]["params"]["device"] == "cpu"
    assert params["first_stage_config"]["target"] == "ldm.models.autoencoder.AutoencoderKL"
    assert params["unet_config"]["params"]["in_channels"] == 12
    assert params["unet_config"]["params"]["out_channels"] == 4
    assert params["cond_stage_trainable"] is False
    assert params["use_ema"] is True


def test_batch_hook_forwards_default_and_explicit_loader_without_changing_batch():
    class LegacyHook:
        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            return batch, batch_idx, dataloader_idx

    class Compatible(Lightning19BatchHook, LegacyHook):
        pass

    batch = {"sentinel": object()}
    model = Compatible()
    output = model.on_train_batch_start(batch, 7)
    assert output[0] is batch
    assert output[1:] == (7, 0)
    assert model.on_train_batch_start(batch, 7, 3)[1:] == (7, 3)


def test_factory_rejects_unverified_lightning_before_importing_upstream(monkeypatch):
    monkeypatch.setattr(diffv2ir_training, "version", lambda name: "2.0.0")
    with pytest.raises(RuntimeError, match="Lightning1.9.5"):
        diffv2ir_training.create_training_model()
