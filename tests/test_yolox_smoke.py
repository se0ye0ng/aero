import json
from pathlib import Path

import pytest

from aero_ir.detect.yolox_smoke import (
    build_yolox_smoke_report,
    parse_yolox_smoke_log,
    verify_yolox_smoke_report,
)
from aero_ir.utils.manifest import file_sha256

LOG = "\n".join(
    [
        "2026-09-05 20:23:48.839 | INFO | args: Namespace("
        "experiment_name='smoke', batch_size=8, devices=1, fp16=True)",
        "2026-09-05 20:25:05.069 | INFO | epoch: 1/1, iter: 10/16, mem: 1812Mb, "
        "iter_time: 7.069s, data_time: 0.012s, total_loss: 17.4, iou_loss: 4.6, "
        "l1_loss: 2.2, conf_loss: 9.5, cls_loss: 1.1, lr: 6.250e-05, size: 640, "
        "ETA: 0:00:42",
        "2026-09-05 20:25:51.589 | INFO | Average forward time: 38.14 ms, "
        "Average NMS time: 4.44 ms, Average inference time: 42.58 ms",
        "2026-09-05 20:25:51.820 | INFO | Training of experiment is done and the best AP is 0.00",
    ]
)


def test_parse_completed_yolox_smoke_log():
    parsed = parse_yolox_smoke_log(LOG)

    assert parsed["completed"]
    assert parsed["batch_size"] == 8
    assert parsed["gradient_accumulation"] == {
        "steps": 1,
        "effective_batch_size": 8,
        "optimizer_steps_per_epoch": 16,
    }
    assert parsed["iterations_per_epoch"] == 16
    assert parsed["logged_cuda_memory_mib"] == 1812
    assert parsed["losses"]["total"] == 17.4
    assert parsed["inference"]["total_ms"] == 42.58
    assert parsed["best_ap"] == 0.0
    assert parsed["wall_time_s"] == pytest.approx(122.981)


def test_parse_yolox_smoke_records_accumulation_policy():
    accumulated_log = LOG.replace(
        "2026-09-05 20:25:05.069 | INFO | epoch:",
        "2026-09-05 20:24:00.000 | INFO | Effective batch: 64 "
        "(microbatch 8 x accumulation 8); optimizer steps/epoch: 2\n"
        "2026-09-05 20:25:05.069 | INFO | epoch:",
    )

    parsed = parse_yolox_smoke_log(accumulated_log)

    assert parsed["gradient_accumulation"] == {
        "steps": 8,
        "effective_batch_size": 64,
        "optimizer_steps_per_epoch": 2,
    }


def test_parse_yolox_smoke_uses_latest_attempt_in_an_appended_log():
    failed_attempt = LOG.split("Average forward time", maxsplit=1)[0]
    appended = failed_attempt + "\n" + LOG

    parsed = parse_yolox_smoke_log(appended)

    assert parsed["wall_time_s"] == pytest.approx(122.981)


def test_parse_yolox_smoke_requires_completion():
    with pytest.raises(ValueError, match="successful completion marker"):
        parse_yolox_smoke_log(LOG.replace("Training of experiment is done", "Training stopped"))


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_smoke_report_binds_inputs_and_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    _write(run_dir / "train_log.txt", LOG)
    for name in ("epoch_1_ckpt.pth", "last_epoch_ckpt.pth", "latest_ckpt.pth"):
        _write(run_dir / name, name)
    _write(run_dir / "tensorboard" / "events.out.tfevents.1.node39.2.0", "event")
    _write(tmp_path / "manifest.json", "manifest")
    _write(tmp_path / "preprocess.json", "preprocess")
    train = tmp_path / "prepared" / "annotations" / "smoke_train.json"
    val = tmp_path / "prepared" / "annotations" / "smoke_val.json"
    _write(train, "train")
    _write(val, "val")
    preflight_path = tmp_path / "prepared" / "preflight.json"
    preflight = {
        "gate": "pass",
        "smoke_annotations": {
            "train": {
                "path": str(train),
                "sha256": file_sha256(train),
                "images": 128,
                "annotations": 12,
            },
            "val": {
                "path": str(val),
                "sha256": file_sha256(val),
                "images": 64,
                "annotations": 6,
            },
        },
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    report = build_yolox_smoke_report(
        run_dir,
        dataset_manifest=tmp_path / "manifest.json",
        preprocess=tmp_path / "preprocess.json",
        prepared_preflight=preflight_path,
    )

    verify_yolox_smoke_report(report)
    assert report["gate"] == "pass"
    assert report["scientific_status"] == "engineering_only_not_reportable"
    assert report["host"] == "node39"
    (run_dir / "latest_ckpt.pth").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_yolox_smoke_report(report)
