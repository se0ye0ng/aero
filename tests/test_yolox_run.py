import json
from pathlib import Path

import pytest

from aero_ir.detect import yolox_run
from aero_ir.detect.yolox_run import (
    execute_yolox_run,
    finalize_yolox_run,
    load_yolox_run_spec,
    prepare_yolox_run_spec,
    save_yolox_run_spec,
    verify_yolox_run_spec,
)
from aero_ir.utils.manifest import canonical_hash, file_sha256, load_manifest, verify_run_manifest


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repository = tmp_path / "repo"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write(repository / "code.py", "fixed = True\n")

    manifest = {
        "schema_version": 1,
        "dataset": "fixture",
        "categories": [],
        "splits": {},
        "representation": "analytics16",
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    preprocess = {
        "schema_version": 1,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "lower_dn": 10,
        "upper_dn": 20,
    }
    preprocess["preprocess_sha256"] = canonical_hash(preprocess)
    preprocess_path = tmp_path / "preprocess.json"
    preprocess_path.write_text(json.dumps(preprocess), encoding="utf-8")

    prepared = tmp_path / "prepared"
    annotations = {}
    for split in ("train", "val"):
        path = prepared / "annotations" / f"{split}.json"
        payload = {
            "info": {"aero_manifest_sha256": manifest["manifest_sha256"]},
            "images": [],
            "annotations": [],
            "categories": [],
        }
        _write(path, json.dumps(payload))
        annotations[split] = {"sha256": file_sha256(path)}
    preflight = {"gate": "pass", "annotations": annotations}
    _write(prepared / "preflight.json", json.dumps(preflight))
    return {
        "repository": repository,
        "dataset": dataset,
        "manifest": manifest_path,
        "preprocess": preprocess_path,
        "prepared": prepared,
        "output": tmp_path / "runs",
    }


def _timing_spec(tmp_path: Path) -> dict:
    paths = _fixture(tmp_path)
    return prepare_yolox_run_spec(
        mode="timing",
        run_id="timing-seed0",
        repository_root=paths["repository"],
        dataset_root=paths["dataset"],
        dataset_manifest=paths["manifest"],
        preprocess=paths["preprocess"],
        prepared_root=paths["prepared"],
        output_root=paths["output"],
        epochs=300,
        max_train_iters=16,
        timing_warmup_iters=8,
        code_paths=("code.py",),
    )


def test_run_spec_is_content_addressed_and_rejects_changed_code(tmp_path):
    spec = _timing_spec(tmp_path)
    path = tmp_path / "spec.json"
    save_yolox_run_spec(spec, path)

    assert load_yolox_run_spec(path)["resolved_config"]["max_train_iters"] == 16

    Path(spec["code"]["code.py"]["path"]).write_text("fixed = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code.py.*hash mismatch"):
        load_yolox_run_spec(path)


def test_timing_spec_requires_normal_augmentation_and_complete_steps(tmp_path):
    paths = _fixture(tmp_path)
    common = {
        "mode": "timing",
        "run_id": "timing",
        "repository_root": paths["repository"],
        "dataset_root": paths["dataset"],
        "dataset_manifest": paths["manifest"],
        "preprocess": paths["preprocess"],
        "prepared_root": paths["prepared"],
        "output_root": paths["output"],
        "code_paths": ("code.py",),
    }
    with pytest.raises(ValueError, match="epochs > 15"):
        prepare_yolox_run_spec(
            **common,
            epochs=1,
            max_train_iters=16,
            timing_warmup_iters=8,
        )
    with pytest.raises(ValueError, match="divisible"):
        prepare_yolox_run_spec(
            **common,
            epochs=300,
            max_train_iters=15,
            timing_warmup_iters=8,
        )


def test_formal_training_spec_requires_clean_git_state(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    common = {
        "mode": "train",
        "run_id": "full-seed0",
        "repository_root": paths["repository"],
        "dataset_root": paths["dataset"],
        "dataset_manifest": paths["manifest"],
        "preprocess": paths["preprocess"],
        "prepared_root": paths["prepared"],
        "output_root": paths["output"],
        "code_paths": ("code.py",),
    }
    monkeypatch.setattr(
        yolox_run,
        "_git_state",
        lambda repository_root: {"sha": "abc123", "dirty": True},
    )
    with pytest.raises(ValueError, match="clean git working tree"):
        prepare_yolox_run_spec(**common)

    monkeypatch.setattr(
        yolox_run,
        "_git_state",
        lambda repository_root: {"sha": "abc123", "dirty": False},
    )
    spec = prepare_yolox_run_spec(**common)
    assert spec["scientific_status"] == "reportable_candidate"
    assert spec["git"] == {"sha": "abc123", "dirty": False}
    verify_yolox_run_spec(spec)

    spec["git"]["dirty"] = True
    unsigned = {key: value for key, value in spec.items() if key != "spec_sha256"}
    spec["spec_sha256"] = canonical_hash(unsigned)
    with pytest.raises(ValueError, match="must identify a clean git"):
        verify_yolox_run_spec(spec)


def test_finalize_timing_run_writes_verifiable_replay_manifest(tmp_path):
    spec = _timing_spec(tmp_path)
    spec_path = tmp_path / "spec.json"
    save_yolox_run_spec(spec, spec_path)
    run_dir = Path(spec["output_root"]) / spec["run_id"]
    _write(run_dir / "train_log.txt", "timing complete\n")
    _write(run_dir / "tensorboard" / "events.out.tfevents.1.node.2.0", "event")
    metrics = {
        "schema_version": 1,
        "kind": "flir_yolox_bounded_full_data_timing",
        "status": "pass",
        "full_dataset_images": 10742,
        "full_iterations_per_epoch": 1343,
        "executed_iterations": 16,
        "executed_optimizer_steps": 2,
        "configured_epochs": 300,
        "executed_epochs": 1,
        "microbatch_size": 8,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 64,
        "normal_augmentation": {"mosaic": True, "mixup": True},
        "all_losses_finite": True,
    }
    metrics["metrics_sha256"] = canonical_hash(metrics)
    _write(run_dir / "timing_metrics.json", json.dumps(metrics))
    runtime = {
        "schema_version": 1,
        "status": "pass",
        "host": "node",
        "python_version": "3.11",
        "torch_version": "2.9.0+cu128",
        "torch_cuda_version": "12.8",
        "device_name": "fixture",
        "peak_cuda_memory_mib": 123.0,
    }
    runtime["runtime_sha256"] = canonical_hash(runtime)
    _write(run_dir / "runtime.json", json.dumps(runtime))

    manifest_path = finalize_yolox_run(
        spec_path,
        run_dir,
        started_at="2026-09-05T00:00:00+00:00",
    )

    manifest = load_manifest(manifest_path)
    assert manifest["metrics"]["normal_augmentation"]["mosaic"]
    assert manifest["replay_metrics_path"] == "timing-seed0/timing_metrics.json"
    assert verify_run_manifest(manifest_path)["ok"]


def test_execute_uses_verified_full_data_environment_without_library_override(
    tmp_path,
    monkeypatch,
):
    spec = _timing_spec(tmp_path)
    spec_path = tmp_path / "spec.json"
    save_yolox_run_spec(spec, spec_path)
    captured = {}

    def fake_run(command, *, cwd, env, check):
        captured.update({"command": command, "cwd": cwd, "env": env, "check": check})

    expected_manifest = tmp_path / "run_manifest.json"
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bad/cuda")
    monkeypatch.setattr(yolox_run.subprocess, "run", fake_run)
    monkeypatch.setattr(
        yolox_run,
        "finalize_yolox_run",
        lambda *args, **kwargs: expected_manifest,
    )

    result = execute_yolox_run(spec_path)

    assert result == expected_manifest
    assert "LD_LIBRARY_PATH" not in captured["env"]
    assert captured["env"]["AERO_YOLOX_RUN_MODE"] == "timing"
    assert captured["env"]["AERO_YOLOX_TRAIN_ANN"] == "train.json"
    assert captured["env"]["AERO_YOLOX_MAX_TRAIN_ITERS"] == "16"
    assert captured["command"][-4:] == [
        "-expn",
        "timing-seed0",
        "-l",
        "tensorboard",
    ]
