#!/usr/bin/env python3
"""Verify and summarize an immutable Anti-UAV300 dense-registration result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_dense_registration import _gate_report


def check_report(
    report_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    stage: str,
    screen_samples: int = 8,
) -> tuple[bool, str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"fine-tuned checkpoint is missing: {checkpoint_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("kind") != "antiuav300_superfusion_dense_registration_audit"
    ):
        raise ValueError(f"unrecognized dense-registration report: {report_path}")
    expected_hash = report.get("dense_registration_audit_sha256")
    unhashed = {
        key: value for key, value in report.items() if key != "dense_registration_audit_sha256"
    }
    if not isinstance(expected_hash, str) or canonical_hash(unhashed) != expected_hash:
        raise ValueError(f"dense-registration report content hash mismatch: {report_path}")
    if report.get("root") != str(dataset_root.resolve()):
        raise ValueError("dense-registration report refers to a different dataset root")
    data_usage = report.get("data_usage")
    frame_selection = (
        f"{screen_samples} endpoint-inclusive usable pairs per sequence"
        if stage == "screen"
        else "all usable paired target frames"
    )
    if not isinstance(data_usage, dict) or data_usage.get("frame_selection") != frame_selection:
        raise ValueError("dense-registration report uses a different frame-selection protocol")
    model = report.get("model")
    if not isinstance(model, dict) or model.get("checkpoint_sha256") != file_sha256(
        checkpoint_path
    ):
        raise ValueError("dense-registration report refers to a different checkpoint")
    gates = report.get("gates")
    metrics = report.get("metrics")
    if not isinstance(gates, dict) or not isinstance(metrics, dict):
        raise ValueError("dense-registration report is missing gates or metrics")
    for split in ("train", "val"):
        split_metrics = metrics.get(split)
        if not isinstance(split_metrics, dict) or "frame_pass_rate" not in split_metrics:
            raise ValueError(f"dense-registration report is missing {split} metrics")
    if gates != _gate_report(metrics, exhaustive=stage == "full"):
        raise ValueError("dense-registration report gates disagree with frozen metrics")
    gate_name = "sequence_balanced_screen" if stage == "screen" else "generator_training_eligible"
    gate = gates.get(gate_name)
    if gate not in {"pass", "hold"}:
        raise ValueError(f"unexpected {stage} gate value: {gate!r}")
    train_rate = metrics["train"]["frame_pass_rate"]["joint"]
    val_rate = metrics["val"]["frame_pass_rate"]["joint"]
    message = (
        f"Existing Anti-UAV300 registration {stage}: {gate.upper()} "
        f"(train joint {train_rate:.2%}, validation joint {val_rate:.2%}; "
        f"frozen requirement 95%). Report: {report_path}"
    )
    if gate == "hold":
        message += "\nGenerator training remains HOLD; the saved result will not be overwritten."
    return gate == "pass", message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "full"), required=True)
    parser.add_argument("--screen-samples", type=int, default=8)
    args = parser.parse_args()
    try:
        passed, message = check_report(
            args.report,
            args.checkpoint,
            args.root,
            args.stage,
            args.screen_samples,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"invalid prior registration result: {error}\n")
    print(message)
    if not passed:
        parser.exit(3)


if __name__ == "__main__":
    main()
