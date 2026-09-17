#!/usr/bin/env python3
"""Verify and summarize an immutable Anti-UAV300 v3 registration result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.audit_antiuav300_bidirectional_registration import gate_report


def check_report(
    report_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    stage: str,
    screen_samples: int = 8,
) -> tuple[bool, str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"v3 checkpoint is missing: {checkpoint_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("kind") != "antiuav300_bidirectional_dense_registration_audit_v3"
    ):
        raise ValueError(f"unrecognized v3 registration report: {report_path}")
    expected_hash = report.get("bidirectional_registration_audit_sha256")
    unhashed = {
        key: value
        for key, value in report.items()
        if key != "bidirectional_registration_audit_sha256"
    }
    if not isinstance(expected_hash, str) or canonical_hash(unhashed) != expected_hash:
        raise ValueError(f"v3 report content hash mismatch: {report_path}")
    if report.get("root") != str(dataset_root.resolve()):
        raise ValueError("v3 report refers to a different dataset root")
    expected_selection = (
        f"{screen_samples} endpoint-inclusive usable pairs per sequence"
        if stage == "screen"
        else "all usable paired target frames"
    )
    if report.get("data_usage", {}).get("frame_selection") != expected_selection:
        raise ValueError("v3 report uses a different frame-selection protocol")
    if report.get("model", {}).get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise ValueError("v3 report refers to a different checkpoint")
    metrics = report.get("metrics")
    gates = report.get("gates")
    if not isinstance(metrics, dict) or not isinstance(gates, dict):
        raise ValueError("v3 report is missing metrics or gates")
    if gates != gate_report(metrics, exhaustive=stage == "full"):
        raise ValueError("v3 report gates disagree with frozen metrics")
    gate_name = "sequence_balanced_screen" if stage == "screen" else "generator_training_eligible"
    gate = gates.get(gate_name)
    if gate not in {"pass", "hold"}:
        raise ValueError(f"unexpected {stage} gate value: {gate!r}")
    train_forward = metrics["train"]["visible_to_infrared_backward_map"]["frame_pass_rate"]["joint"]
    train_reverse = metrics["train"]["infrared_to_visible_backward_map"]["frame_pass_rate"]["joint"]
    val_forward = metrics["val"]["visible_to_infrared_backward_map"]["frame_pass_rate"]["joint"]
    val_reverse = metrics["val"]["infrared_to_visible_backward_map"]["frame_pass_rate"]["joint"]
    message = (
        f"Existing Anti-UAV300 bidirectional v3 {stage}: {gate.upper()} "
        f"(train forward/reverse {train_forward:.2%}/{train_reverse:.2%}; "
        f"validation forward/reverse {val_forward:.2%}/{val_reverse:.2%}; "
        "frozen requirement 95% each). "
        f"Report: {report_path}"
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
        parser.exit(2, f"invalid prior v3 registration result: {error}\n")
    print(message)
    if not passed:
        parser.exit(3)


if __name__ == "__main__":
    main()
