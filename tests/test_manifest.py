import json
import sys

import pytest

from aero_ir.utils import manifest as manifest_module
from aero_ir.utils.manifest import (
    RunManifest,
    canonical_hash,
    file_sha256,
    load_manifest,
    verify_run_manifest,
)


def test_run_manifest_detects_tampering_and_artifact_changes(tmp_path):
    artifact = tmp_path / "dataset.json"
    artifact.write_text('{"fixed": true}\n', encoding="utf-8")
    config = {"experiment": "fixture", "seed": 3}
    run = RunManifest(
        run_id="fixture-3",
        config_hash=canonical_hash(config),
        experiment="fixture",
        seed=3,
        resolved_config=config,
        artifacts={"dataset_manifest": {"path": artifact.name, "sha256": file_sha256(artifact)}},
    )
    manifest_path = tmp_path / "run.json"
    run.save(manifest_path)

    assert verify_run_manifest(manifest_path)["ok"]

    artifact.write_text('{"fixed": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact.*hash mismatch"):
        verify_run_manifest(manifest_path)

    artifact.write_text('{"fixed": true}\n', encoding="utf-8")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["seed"] = 4
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="run manifest hash mismatch"):
        load_manifest(manifest_path)


def test_run_manifest_replays_and_compares_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    command = [
        sys.executable,
        "-c",
        ("from pathlib import Path; Path('metrics.json').write_text('{\\\"map_50\\\": 0.75}')"),
    ]
    config = {"experiment": "fixture", "seed": 0}
    run = RunManifest(
        run_id="fixture-replay",
        config_hash=canonical_hash(config),
        experiment="fixture",
        seed=0,
        resolved_config=config,
        command=command,
        working_directory=str(tmp_path),
        metrics_path=metrics_path.name,
        metrics={"map_50": 0.75},
    )
    manifest_path = tmp_path / "run.json"
    run.save(manifest_path)

    report = verify_run_manifest(manifest_path, execute=True)

    assert report["ok"]
    assert report["replayed"]
    assert report["metric_failures"] == []


def test_run_manifest_replays_into_an_isolated_output_root(tmp_path):
    command = [
        sys.executable,
        "-c",
        (
            "import os; from pathlib import Path; "
            "p=Path(os.environ['AERO_REPLAY_OUTPUT_ROOT'])/'run'/'metrics.json'; "
            'p.parent.mkdir(parents=True); p.write_text(\'{"status": "pass"}\')'
        ),
    ]
    config = {"experiment": "isolated-replay"}
    run = RunManifest(
        run_id="isolated-replay",
        config_hash=canonical_hash(config),
        experiment="fixture",
        seed=0,
        resolved_config=config,
        command=command,
        working_directory=str(tmp_path),
        metrics_path="original.json",
        replay_metrics_path="run/metrics.json",
        metrics={"status": "pass"},
        command_environment={"REMOVE_ME": "set"},
        unset_environment=["REMOVE_ME", "LD_LIBRARY_PATH"],
    )
    manifest_path = tmp_path / "run.json"
    run.save(manifest_path)

    report = verify_run_manifest(manifest_path, execute=True)

    assert report["ok"]
    assert report["replayed"]
    assert not (tmp_path / "run" / "metrics.json").exists()


def test_historical_git_blob_verifies_but_cannot_execute(tmp_path, monkeypatch):
    artifact = tmp_path / "code.py"
    artifact.write_text("old code\n", encoding="utf-8")
    expected = file_sha256(artifact)
    config = {"experiment": "historical"}
    run = RunManifest(
        run_id="historical",
        config_hash=canonical_hash(config),
        experiment="fixture",
        seed=0,
        resolved_config=config,
        artifacts={"input__code.py": {"path": "code.py", "sha256": expected}},
        working_directory=str(tmp_path),
        git_sha="a" * 40,
    )
    manifest_path = tmp_path / "run.json"
    run.save(manifest_path)
    artifact.write_text("new code\n", encoding="utf-8")
    monkeypatch.setattr(manifest_module, "_git_blob_sha256", lambda *args: expected)

    report = verify_run_manifest(manifest_path)

    assert report["ok"]
    assert report["verified_from_git"] == ["input__code.py"]
    with pytest.raises(ValueError, match="checked-out source files"):
        verify_run_manifest(manifest_path, execute=True)
