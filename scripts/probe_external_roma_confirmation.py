"""Run the unchanged RoMa scorer/model on frozen same-scene confirmation pairs."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from aero_ir.utils.manifest import file_sha256
from scripts import probe_external_roma as runner
from scripts.probe_external_boundary_confirmation import CONFIG, CONFIG_SHA, MANIFEST_SHA

ORIGINAL_PROTOCOL = runner.protocol
RUNNER_SHA = "0f1d3c93b3f1ee5a6a4886a733b0076a9b4ad3d8dbf23d56d91b364e7be68118"
DEVELOPMENT = Path("experiments/registration_external_roma_gpu_01/report.json")
DEVELOPMENT_SHA = "b727e3aeea2cd03dc158a759590de3a9687fbe85aa8474ca17be644105fdb5f9"


def panel():
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("confirmation policy changed")
    config = yaml.safe_load(CONFIG.read_text())
    root = Path(config["cache"])
    manifest_path = root / "manifest.json"
    if file_sha256(manifest_path) != MANIFEST_SHA:
        raise ValueError("confirmation manifest changed")
    manifest = json.loads(manifest_path.read_text())
    pairs = [
        f"{scene}/{i}" for scene in config["scenes"] for i in config["confirmation_sample_ids"]
    ]
    if len(pairs) != 12 or len(set(pairs)) != 12 or not set(pairs).issubset(manifest["pairs"]):
        raise ValueError("incomplete confirmation panel")
    files = {row["path"]: row["sha256"] for row in manifest["files"]}
    duplicates = [
        p
        for p in pairs
        if all(
            files[f"{p}/{name}"] == files[f"{p.split('/')[0]}/1/{name}"]
            for name in ("points_rgb.txt", "points_thermal.txt")
        )
    ]
    return root, pairs, files, duplicates


def protocol():
    if file_sha256(runner.__file__) != RUNNER_SHA or file_sha256(DEVELOPMENT) != DEVELOPMENT_SHA:
        raise ValueError("development model/scorer or evidence changed")
    _, _, hashes = ORIGINAL_PROTOCOL()
    root, pairs, files, _ = panel()
    hashes.update({str(root / name): digest for name, digest in files.items()})
    for path in (
        CONFIG,
        root / "manifest.json",
        DEVELOPMENT,
        Path(__file__),
        Path("scripts/run_external_roma_confirmation_gpu.sh"),
        Path("scripts/probe_external_boundary_confirmation.py"),
    ):
        hashes[str(path)] = file_sha256(path)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed input: {path}")
    return root, pairs, hashes


def aggregate(rows, duplicates):
    result = {}
    for subset in ("all12", "without_development_duplicate_labels"):
        selected = rows if subset == "all12" else [r for r in rows if r["pair"] not in duplicates]
        variants = {}
        for variant in ("fixed_confidence_ge_half", "ungated_diagnostic"):
            summaries = [r["scores"][variant][0]["summary"] for r in selected]
            variants[variant] = dict(
                landmarks=sum(s["landmarks"] for s in summaries),
                unavailable=sum(s["unprojectable"] for s in summaries),
                macro_pck={
                    str(t): float(np.mean([s["pck_all_landmarks"][str(t)] for s in summaries]))
                    for t in (1.0, 3.0, 5.0, 10.0)
                },
            )
        result[subset] = dict(pairs=len(selected), target="thermal", variants=variants)
    return result


def main():
    # Replace only the input-panel provider, never the frozen inference or scoring code.
    action = sys.argv[1]
    out = Path(sys.argv[sys.argv.index("--out-dir") + 1])
    with patch.object(runner, "protocol", protocol):
        runner.main()
    if action == "preflight":
        return
    report_path = out / "report.json"
    report = json.loads(report_path.read_text())
    _, _, _, duplicates = panel()
    summary = aggregate(report["rows"], duplicates)
    if action == "run":
        report.update(
            aggregate=summary,
            duplicate_development_labels=duplicates,
            limitations=[
                "Same-scene confirmation, not unseen-scene generalization.",
                "Duplicate development labels also reported separately excluded.",
                "Ungated diagnostic is not a replacement qualification gate.",
                "External results do not qualify Anti-UAV300 or approve generator training.",
            ],
        )
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    elif summary != report["aggregate"] or duplicates != report["duplicate_development_labels"]:
        raise ValueError("confirmation aggregate differs")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
