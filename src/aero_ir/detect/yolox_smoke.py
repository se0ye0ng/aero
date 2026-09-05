"""Validation and content-addressing for YOLOX engineering-smoke runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from aero_ir.utils.manifest import canonical_hash, file_sha256

SMOKE_REPORT_SCHEMA = 1

_ARGUMENTS = re.compile(
    r"experiment_name='(?P<experiment>[^']+)'.*?batch_size=(?P<batch>\d+).*?"
    r"devices=(?P<devices>\d+).*?fp16=(?P<fp16>True|False)"
)
_ITERATION = re.compile(
    r"epoch: (?P<epoch>\d+)/(?P<epochs>\d+), iter: (?P<iteration>\d+)/(?P<iterations>\d+), "
    r"mem: (?P<memory>[\d.]+)Mb, iter_time: (?P<iteration_s>[\d.]+)s, "
    r"data_time: (?P<data_s>[\d.]+)s, total_loss: (?P<total_loss>[\d.]+), "
    r"iou_loss: (?P<iou_loss>[\d.]+), l1_loss: (?P<l1_loss>[\d.]+), "
    r"conf_loss: (?P<conf_loss>[\d.]+), cls_loss: (?P<cls_loss>[\d.]+), "
    r"lr: (?P<lr>[\d.eE+-]+), size: (?P<input_size>\d+)"
)
_INFERENCE = re.compile(
    r"Average forward time: (?P<forward_ms>[\d.]+) ms, "
    r"Average NMS time: (?P<nms_ms>[\d.]+) ms, "
    r"Average inference time: (?P<inference_ms>[\d.]+) ms"
)
_COMPLETION = re.compile(r"Training of experiment is done and the best AP is (?P<best_ap>[\d.]+)")
_EFFECTIVE_BATCH = re.compile(
    r"Effective batch: (?P<effective>\d+) \(microbatch (?P<microbatch>\d+) x "
    r"accumulation (?P<accumulation>\d+)\); optimizer steps/epoch: (?P<optimizer_steps>\d+)"
)
_TIMESTAMP = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")


def _required_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"YOLOX log has no {label}")
    return match


def parse_yolox_smoke_log(text: str) -> dict:
    """Extract pass/fail evidence from one completed upstream YOLOX smoke log."""
    completion_matches = list(_COMPLETION.finditer(text))
    if not completion_matches:
        raise ValueError("YOLOX log has no successful completion marker")
    final_completion = completion_matches[-1]
    argument_matches = [
        match for match in _ARGUMENTS.finditer(text) if match.start() < final_completion.start()
    ]
    if not argument_matches:
        raise ValueError("YOLOX log has no training arguments")
    arguments = argument_matches[-1]
    attempt_start = text.rfind("\n", 0, arguments.start()) + 1
    attempt_end = text.find("\n", final_completion.end())
    attempt = text[attempt_start : attempt_end if attempt_end >= 0 else len(text)]
    iterations = list(_ITERATION.finditer(attempt))
    if not iterations:
        raise ValueError("YOLOX log has no training iteration metrics")
    inference = _required_match(_INFERENCE, attempt, "inference timing")
    completion = _required_match(_COMPLETION, attempt, "successful completion marker")

    timestamp_lines = [
        match.group("timestamp")
        for line in attempt.splitlines()
        if (match := _TIMESTAMP.match(line)) is not None
    ]
    if len(timestamp_lines) < 2:
        raise ValueError("YOLOX log has insufficient timestamps")
    started = datetime.fromisoformat(timestamp_lines[0])
    completed = datetime.fromisoformat(timestamp_lines[-1])
    if completed < started:
        raise ValueError("YOLOX log completion precedes its start")

    last_iteration = iterations[-1].groupdict()
    batch_size = int(arguments.group("batch"))
    effective_batch = _EFFECTIVE_BATCH.search(attempt)
    if effective_batch is None:
        accumulation = {
            "steps": 1,
            "effective_batch_size": batch_size,
            "optimizer_steps_per_epoch": int(last_iteration["iterations"]),
        }
    else:
        if int(effective_batch.group("microbatch")) != batch_size:
            raise ValueError("logged effective-batch microbatch differs from CLI batch")
        accumulation = {
            "steps": int(effective_batch.group("accumulation")),
            "effective_batch_size": int(effective_batch.group("effective")),
            "optimizer_steps_per_epoch": int(effective_batch.group("optimizer_steps")),
        }

    return {
        "completed": True,
        "experiment_name": arguments.group("experiment"),
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "devices": int(arguments.group("devices")),
        "fp16": arguments.group("fp16") == "True",
        "epoch": int(last_iteration["epoch"]),
        "epochs": int(last_iteration["epochs"]),
        "last_logged_iteration": int(last_iteration["iteration"]),
        "iterations_per_epoch": int(last_iteration["iterations"]),
        "logged_cuda_memory_mib": float(last_iteration["memory"]),
        "iteration_time_s": float(last_iteration["iteration_s"]),
        "data_time_s": float(last_iteration["data_s"]),
        "losses": {
            "total": float(last_iteration["total_loss"]),
            "iou": float(last_iteration["iou_loss"]),
            "l1": float(last_iteration["l1_loss"]),
            "confidence": float(last_iteration["conf_loss"]),
            "classification": float(last_iteration["cls_loss"]),
        },
        "learning_rate": float(last_iteration["lr"]),
        "input_size": int(last_iteration["input_size"]),
        "inference": {
            "forward_ms": float(inference.group("forward_ms")),
            "nms_ms": float(inference.group("nms_ms")),
            "total_ms": float(inference.group("inference_ms")),
        },
        "best_ap": float(completion.group("best_ap")),
        "started_at_local": timestamp_lines[0],
        "completed_at_local": timestamp_lines[-1],
        "wall_time_s": (completed - started).total_seconds(),
    }


def _artifact(path: Path) -> dict[str, str | int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = path.resolve()
    try:
        recorded_path = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        recorded_path = resolved
    return {
        "path": str(recorded_path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def build_yolox_smoke_report(
    run_dir: str | Path,
    *,
    dataset_manifest: str | Path,
    preprocess: str | Path,
    prepared_preflight: str | Path,
) -> dict:
    """Validate a smoke run and bind it to its exact data and output artifacts."""
    run_dir = Path(run_dir)
    log_path = run_dir / "train_log.txt"
    training = parse_yolox_smoke_log(log_path.read_text(encoding="utf-8"))
    preflight_path = Path(prepared_preflight)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("gate") != "pass":
        raise ValueError("FLIR YOLOX preflight gate did not pass")

    smoke_inputs: dict[str, dict[str, str | int]] = {}
    for split in ("train", "val"):
        specification = preflight.get("smoke_annotations", {}).get(split, {})
        annotation_path = Path(str(specification.get("path", "")))
        artifact = _artifact(annotation_path)
        if artifact["sha256"] != specification.get("sha256"):
            raise ValueError(f"{split} smoke annotation hash differs from preflight")
        artifact["images"] = int(specification["images"])
        artifact["annotations"] = int(specification["annotations"])
        smoke_inputs[split] = artifact

    checkpoints = {
        name: _artifact(run_dir / name)
        for name in ("epoch_1_ckpt.pth", "last_epoch_ckpt.pth", "latest_ckpt.pth")
    }
    event_files = sorted((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError(f"expected one TensorBoard event file, found {len(event_files)}")
    event_file = event_files[0]
    event_parts = event_file.name.split(".")
    host = event_parts[4] if len(event_parts) >= 6 else "unknown"

    report = {
        "schema_version": SMOKE_REPORT_SCHEMA,
        "kind": "flir_yolox_engineering_smoke",
        "gate": "pass",
        "scientific_status": "engineering_only_not_reportable",
        "interpretation": (
            "This one-epoch, 128/64-image scratch run validates execution only. "
            "Its AP is not an E1 result."
        ),
        "host": host,
        "training": training,
        "inputs": {
            "dataset_manifest": _artifact(Path(dataset_manifest)),
            "preprocess": _artifact(Path(preprocess)),
            "prepared_preflight": _artifact(preflight_path),
            "smoke_annotations": smoke_inputs,
        },
        "outputs": {
            "train_log": _artifact(log_path),
            "tensorboard": _artifact(event_file),
            "checkpoints": checkpoints,
        },
    }
    report["report_sha256"] = canonical_hash(report)
    return report


def verify_yolox_smoke_report(report: dict) -> None:
    """Reject a modified report or any changed artifact bound into it."""
    if report.get("schema_version") != SMOKE_REPORT_SCHEMA:
        raise ValueError("unsupported YOLOX smoke report schema")
    recorded = report.get("report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if not isinstance(recorded, str) or canonical_hash(unsigned) != recorded:
        raise ValueError("YOLOX smoke report hash mismatch")

    artifacts = [
        report["inputs"]["dataset_manifest"],
        report["inputs"]["preprocess"],
        report["inputs"]["prepared_preflight"],
        *report["inputs"]["smoke_annotations"].values(),
        report["outputs"]["train_log"],
        report["outputs"]["tensorboard"],
        *report["outputs"]["checkpoints"].values(),
    ]
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_file() or file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"YOLOX smoke artifact changed: {path}")
