"""CPU interface tests. Fake samplers are not diffusion/registration experiments."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aero_ir.generate import diffv2ir_sampling as sampling  # noqa: E402
from aero_ir.generate.diffv2ir_adapter import DiffV2IRGenerator  # noqa: E402
from aero_ir.generate.diffv2ir_sampling import (  # noqa: E402
    restore_prediction,
    sample_seed,
    sampling_inputs,
)
from aero_ir.utils.manifest import file_sha256  # noqa: E402
from scripts import sample_diffv2ir as worker  # noqa: E402


@pytest.mark.parametrize("paths", [None, [], ["/first/taming", "/second/taming"]])
def test_missing_or_ambiguous_taming_namespace_rejected(monkeypatch, paths):
    spec = None if paths is None else SimpleNamespace(submodule_search_locations=paths)
    monkeypatch.setattr(sampling.importlib.util, "find_spec", lambda name: spec)
    with pytest.raises(ValueError, match="one pinned editable"):
        sampling.taming_source_identity()


def test_taming_namespace_requires_its_own_pinned_revision(tmp_path, monkeypatch):
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path / "taming")])
    monkeypatch.setattr(sampling.importlib.util, "find_spec", lambda name: spec)

    def check_identity(root, *, expected_revision, label):
        assert root == tmp_path.resolve()
        assert expected_revision == sampling.TAMING_REVISION
        assert label == "taming-transformers"
        return {"revision": expected_revision, "tracked_sha256": {"setup.py": "fixture"}}

    monkeypatch.setattr(sampling, "source_identity", check_identity)
    result = sampling.taming_source_identity()
    assert result["root"] == str(tmp_path.resolve())
    assert result["revision"] == sampling.TAMING_REVISION


@pytest.mark.parametrize("shape", [(64, 128), (71, 129), (1, 1), (256, 256)])
def test_padding_restores_original_pixels_without_coordinate_change(shape):
    rgb = np.random.default_rng(5).integers(0, 256, (*shape, 3), dtype=np.uint8)
    before = rgb.copy()
    image, mask, geometry = sampling_inputs(rgb, rgb.copy())
    assert torch.equal(image, mask)
    assert image.shape[-2] % 64 == image.shape[-1] % 64 == 0
    assert np.array_equal(restore_prediction(image, geometry), rgb)
    assert np.array_equal(rgb, before)
    assert geometry["input_size_hw"] == list(shape)


@pytest.mark.parametrize("defect", ["float", "different_size", "gray", "empty"])
def test_invalid_conditioning_rejected(defect):
    rgb = np.zeros((16, 16, 3), np.uint8)
    seg = rgb.copy()
    if defect == "float":
        rgb = rgb.astype(np.float32)
    elif defect == "different_size":
        seg = seg[:10]
    elif defect == "gray":
        seg = seg[..., 0]
    else:
        rgb = rgb[:0]
        seg = seg[:0]
    with pytest.raises(ValueError):
        sampling_inputs(rgb, seg)


def test_invalid_predictions_never_turn_into_uint8_success():
    image = np.zeros((16, 16, 3), np.uint8)
    tensor, _, geometry = sampling_inputs(image, image)
    with pytest.raises(ValueError, match="nonfinite"):
        restore_prediction(tensor * float("nan"), geometry)
    with pytest.raises(ValueError):
        restore_prediction(tensor[:, :, :16], geometry)


def test_per_source_seed_does_not_depend_on_list_order():
    assert sample_seed(0, "a") == sample_seed(0, "a")
    assert sample_seed(0, "a") != sample_seed(1, "a")
    assert sample_seed(0, "a") != sample_seed(0, "b")
    assert 0 <= sample_seed(0, "a") < 2**63
    with pytest.raises(ValueError):
        sample_seed(True, "a")


class FakeSampler:
    """Identity pixels exclusively to test transport; NEVER an IR model."""

    source = {"revision": "test-double", "tracked_sha256": {}}
    taming_source = {"root": "test-double", "revision": "test-double", "tracked_sha256": {}}
    clip_files = {}
    resolved_config = {"test_double": True}

    def __init__(self, **kwargs):
        pass

    def sample(self, rgb, seg, prompt, **kwargs):
        image, _, geometry = sampling_inputs(rgb, seg)
        return restore_prediction(image, geometry), geometry


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    checkpoint = tmp_path / "test_double.ckpt"
    checkpoint.write_bytes(b"not a model; test transport only")
    generator = DiffV2IRGenerator(
        str(checkpoint),
        runtime_python=sys.executable,
        checkpoint_sha256=file_sha256(checkpoint),
        clip_root=str(tmp_path / "fake_clip"),
    )
    source = np.full((67, 99, 3), 173, np.uint8)
    labels = [{"boxes": [[1, 2, 3, 4]], "labels": ["uav"]}]
    kwargs = {
        "segmentations": [source.copy()],
        "prompts": ["frozen fixture prompt"],
        "source_ids": ["train/frame000"],
        "output_dir": tmp_path / "outputs",
        "caption_spec_sha256": "a" * 64,
        "segmentation_spec_sha256": "b" * 64,
    }
    monkeypatch.setattr(worker, "OfficialDiffV2IRSampler", FakeSampler)
    monkeypatch.setattr(worker, "source_identity", lambda root: FakeSampler.source)
    monkeypatch.setattr(worker, "taming_source_identity", lambda: FakeSampler.taming_source)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU-test-double-not-GPU")

    def fake_process(command, *, cwd, env, stdout, stderr, check):
        assert command[0] == str(Path(sys.executable).absolute())
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert env.get("CUDA_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES")
        assert "LD_LIBRARY_PATH" not in env
        # Restore environment after calling the worker in-process in this test.
        with monkeypatch.context() as context:
            context.setenv("HF_HUB_OFFLINE", "1")
            context.setenv("TRANSFORMERS_OFFLINE", "1")
            worker.run(Path(command[-3]), command[-1])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_process)
    return generator, [source], labels, kwargs


def test_adapter_worker_transport_persists_bound_outputs_and_candidate_labels(bridge):
    generator, images, labels, kwargs = bridge
    outputs, copied, provenance = generator.generate(images, labels, **kwargs)
    assert np.array_equal(outputs[0], images[0])
    assert copied == labels and copied is not labels and copied[0] is not labels[0]
    assert provenance[0]["label_validity"] == "unverified_requires_label_audit"
    assert provenance[0]["generator_training_eligible"] == "hold_not_qualified"
    output = kwargs["output_dir"]
    result = json.loads((output / "result.json").read_text())
    assert result["request_sha256"] == file_sha256(output / "request.json")
    assert result["output_radiometry"] == "display_rgb_not_calibrated_radiance"
    assert result["taming_upstream"] == FakeSampler.taming_source
    assert (output / "worker.log").exists()
    with pytest.raises(FileExistsError):
        generator.generate(images, labels, **kwargs)


def test_worker_failure_preserves_logs_and_never_returns_fallback_images(bridge, monkeypatch):
    generator, images, labels, kwargs = bridge
    monkeypatch.setattr(subprocess, "run", lambda *args, **kw: subprocess.CompletedProcess(args, 7))
    with pytest.raises(RuntimeError, match="worker failed.*7"):
        generator.generate(images, labels, **kwargs)
    assert (kwargs["output_dir"] / "request.json").exists()
    assert not (kwargs["output_dir"] / "result.json").exists()


def test_editable_dependency_drift_never_publishes_success(bridge, monkeypatch):
    generator, images, labels, kwargs = bridge
    monkeypatch.setattr(worker, "taming_source_identity", lambda: {"revision": "changed"})
    with pytest.raises(ValueError, match="taming-transformers source changed"):
        generator.generate(images, labels, **kwargs)
    assert (kwargs["output_dir"] / "generated/000000.png").exists()
    assert not (kwargs["output_dir"] / "result.json").exists()


@pytest.mark.parametrize("defect", ["ids", "length", "prompt", "digest", "checkpoint", "batch"])
def test_bad_requests_do_not_create_outputs(bridge, defect):
    generator, images, labels, kwargs = bridge
    if defect == "ids":
        kwargs["source_ids"] = [""]
    elif defect == "length":
        kwargs["prompts"] = []
    elif defect == "prompt":
        kwargs["prompts"] = [""]
    elif defect == "digest":
        kwargs["caption_spec_sha256"] = "not a digest"
    elif defect == "checkpoint":
        generator.checkpoint_sha256 = "0" * 64
    else:
        generator.batch_size = 8
    with pytest.raises(ValueError):
        generator.generate(images, labels, **kwargs)
    assert not kwargs["output_dir"].exists()


def test_changed_result_image_is_rejected(bridge, monkeypatch):
    generator, images, labels, kwargs = bridge
    original = subprocess.run

    def tamper(*args, **kw):
        result = original(*args, **kw)
        (kwargs["output_dir"] / "generated/000000.png").write_bytes(b"tampered")
        return result

    monkeypatch.setattr(subprocess, "run", tamper)
    with pytest.raises(ValueError, match="hash mismatch"):
        generator.generate(images, labels, **kwargs)


@pytest.mark.parametrize("defect", ["kind", "qualification", "labels", "request"])
def test_result_cannot_silently_change_contract(bridge, monkeypatch, defect):
    generator, images, labels, kwargs = bridge
    original = subprocess.run

    def tamper(*args, **kw):
        completed = original(*args, **kw)
        path = kwargs["output_dir"] / "result.json"
        result = json.loads(path.read_text())
        if defect == "kind":
            result["kind"] = "some_other_result"
        elif defect == "qualification":
            result["generator_training_eligible"] = "pass"
        elif defect == "labels":
            result["outputs"][0]["label_validity"] = "pass"
        else:
            result["request_sha256"] = "0" * 64
        path.write_text(json.dumps(result))
        return completed

    monkeypatch.setattr(subprocess, "run", tamper)
    with pytest.raises(ValueError):
        generator.generate(images, labels, **kwargs)
