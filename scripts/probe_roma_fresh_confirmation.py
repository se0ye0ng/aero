"""Frozen RoMa evaluation on IDs 5 and 6, excluding missing-reference Seaside/6."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from aero_ir.utils.manifest import file_sha256
from scripts import probe_external_roma as runner
from scripts.probe_external_roma_confirmation import RUNNER_SHA

CONFIG = Path("configs/experiment/registration_roma_fresh_confirmation.yaml")
CONFIG_SHA = "7582d8db0114c0bdd09ccbe25981c79715ab784d67d9b452810907f6fa9ac4b1"
MANIFEST_SHA = "49087de5a28da91eeea1443a4da09bd9857c1f6a0b1c27f8a692e5d3d142644e"
ORIGINAL_PROTOCOL = runner.protocol


def duplicate_labels(pairs, previous, files):
    def identity(pair):
        return tuple(files[f"{pair}/{name}"] for name in ("points_rgb.txt", "points_thermal.txt"))

    seen = {identity(p) for p in previous}
    duplicates = []
    for pair in pairs:
        key = identity(pair)
        if key in seen:
            duplicates.append(pair)
        seen.add(key)
    return duplicates


def panel():
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("frozen policy changed")
    config = yaml.safe_load(CONFIG.read_text())
    root = Path(config["cache"])
    if file_sha256(root / "manifest.json") != MANIFEST_SHA:
        raise ValueError("frozen download inventory changed")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["config_sha256"] != CONFIG_SHA or manifest["commit"] != config["commit"]:
        raise ValueError("download policy differs")
    pairs = [
        f"{s}/{i}"
        for s in config["scenes"]
        for i in config["evaluation_sample_ids"]
        if f"{s}/{i}" not in config["excluded_missing_reference"]
    ]
    previous = [
        f"{s}/{i}" for s in config["scenes"] for i in config["previously_observed_sample_ids"]
    ]
    if (
        len(pairs) != config["expected_evaluation_pairs"]
        or set(pairs) & set(previous)
        or not set(pairs + previous).issubset(manifest["pairs"])
    ):
        raise ValueError("invalid panel inventory")
    files = {r["path"]: r["sha256"] for r in manifest["files"]}
    return root, pairs, files, duplicate_labels(pairs, previous, files)


def protocol():
    if file_sha256(runner.__file__) != RUNNER_SHA:
        raise ValueError("RoMa model/scorer changed")
    _, _, hashes = ORIGINAL_PROTOCOL()
    root, pairs, files, _ = panel()
    hashes.update({str(root / p): h for p, h in files.items()})
    for path in (
        CONFIG,
        root / "manifest.json",
        Path(__file__),
        Path("scripts/probe_external_roma_confirmation.py"),
        Path("scripts/run_roma_fresh_confirmation_gpu.sh"),
        Path("scripts/fetch_roma_fresh.py"),
    ):
        hashes[str(path)] = file_sha256(path)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed input: {path}")
    return root, pairs, hashes


def aggregate(rows, duplicates):
    result = {}
    for label in ("all_eligible", "without_duplicate_labels"):
        chosen = (
            rows if label == "all_eligible" else [r for r in rows if r["pair"] not in duplicates]
        )
        variants = {}
        for variant in ("fixed_confidence_ge_half", "ungated_diagnostic"):
            summaries = [r["scores"][variant][0]["summary"] for r in chosen]
            variants[variant] = dict(
                landmarks=sum(s["landmarks"] for s in summaries),
                unavailable=sum(s["unprojectable"] for s in summaries),
                macro_pck={
                    str(t): float(np.mean([s["pck_all_landmarks"][str(t)] for s in summaries]))
                    if summaries
                    else None
                    for t in (1.0, 3.0, 5.0, 10.0)
                },
            )
        result[label] = dict(pairs=len(chosen), target="thermal", variants=variants)
    return result


def main():
    action = sys.argv[1]
    out = Path(sys.argv[sys.argv.index("--out-dir") + 1])
    with patch.object(runner, "protocol", protocol):
        runner.main()
    if action == "preflight":
        return
    path = out / "report.json"
    report = json.loads(path.read_text())
    _, _, _, duplicates = panel()
    summary = aggregate(report["rows"], duplicates)
    if action == "run":
        report.update(
            aggregate=summary,
            duplicate_labels=duplicates,
            limitations=[
                "Seaside/6 excluded before scoring because authored landmark files are absent.",
                "New IDs from previously observed scenes, not unseen-scene validation.",
                "No fitting or threshold tuning on these landmarks.",
                "Duplicate labels excluded in secondary analysis only.",
                "External landmark accuracy does not qualify Anti-UAV or dense generation pairs.",
            ],
        )
        path.write_text(json.dumps(report, indent=2, allow_nan=False))
    elif summary != report["aggregate"] or duplicates != report["duplicate_labels"]:
        raise ValueError("aggregate or duplicate inventory changed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
