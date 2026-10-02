"""Synthetic manifests only: never write these fixtures into experiments/."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.verify_antiuav_tiled_matching as verifier
from aero_ir.utils.manifest import file_sha256


@pytest.fixture
def report_fixture(tmp_path, monkeypatch):
    config_path = "configs/experiment/registration_rgb_resolution_cpu.yaml"
    config = dict(
        baseline="fixture/baseline.json",
        baseline_sha256="fixture-digest",
        source_report="fixture/panel.json",
        local_config="fixture/settings.json",
        box_iou_threshold=0.6,
    )
    required = {
        p: "fixture-digest"
        for p in (
            config_path,
            config["baseline"],
            config["source_report"],
            config["local_config"],
            "scripts/probe_antiuav_tiled_matching.py",
            "scripts/run_antiuav_tiled_matching_gpu.sh",
            "src/aero_ir/registration/tiled_matching.py",
        )
    }
    box = [[2.0, 2.0, 5.0, 5.0], [2.0, 2.0, 5.0, 5.0]]
    cases = [
        dict(
            sequence_id=f"fixture{i}", frame_index=0, condition="input_header_crop", boxes_xyxy=box
        )
        for i in range(16)
    ]
    source = dict(
        rows=cases,
        inputs=[
            dict(
                sequence_id=x["sequence_id"],
                inputs={name: dict(resized_shape=[20, 20]) for name in ("visible", "infrared")},
            )
            for x in cases
        ],
    )
    virtual = {
        config_path: json.dumps(config),
        config["baseline"]: json.dumps(dict(input_and_source_sha256=required)),
        config["source_report"]: json.dumps(source),
        config["local_config"]: json.dumps(dict(minimum_controls=6, trim_fraction=0.75)),
    }
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        return virtual[str(path)] if str(path) in virtual else original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        verifier,
        "file_sha256",
        lambda p: "fixture-digest" if str(p) in required else file_sha256(p),
    )
    flags = dict(
        tile_pairs=81,
        before_exact_dedup=0,
        records=[dict(tile0=i, tile1=j, retained=0) for i in range(9) for j in range(9)],
    )
    fit = dict(input_matches=0, unique_rgb_matches=0, status="insufficient_matches")
    score = dict(
        directions=[dict(matches=0, fit=fit, iou=None) for _ in range(2)], joint_pass=False
    )
    names = ("points0", "points1", "confidence", "reverse0", "reverse1", "reverse_confidence")
    arrays = {
        prefix + n: np.empty(0) if "confidence" in n else np.empty((0, 2))
        for prefix in ("", "baseline_")
        for n in names
    }
    rows = []
    for i, case in enumerate(cases):
        artifact = tmp_path / f"{i}.npz"
        np.savez_compressed(artifact, **arrays)
        rows.append(
            dict(
                sequence_id=case["sequence_id"],
                frame_index=0,
                boxes_xyxy=box,
                sizes=[[20, 20], [20, 20]],
                flags=[flags, flags],
                scores=dict(baseline=score, tiled=score),
                artifact=artifact.name,
                sha256=file_sha256(artifact),
            )
        )
    control = tmp_path / "control.npz"
    np.savez_compressed(
        control, points0=np.empty((0, 2)), points1=np.empty((0, 2)), confidence=np.empty(0)
    )
    controls = [
        dict(
            kind=kind,
            artifact=control.name,
            sha256=file_sha256(control),
            matches=0,
            flags=flags,
            pck3=None,
        )
        for kind in ("same_modality_known_shift", "unrelated_pair")
    ]
    report = dict(
        registration_qualified=False,
        generator_training_approved=False,
        input_and_source_sha256=required,
        rows=rows,
        controls=controls,
        summary=dict(baseline=0, tiled=0),
    )
    # JSON round trip removes fixture-only object aliasing before mutations.
    return tmp_path / "report.json", json.loads(json.dumps(report))


def test_all_failure_panel_is_valid_evidence_not_qualification(report_fixture):
    path, report = report_fixture
    path.write_text(json.dumps(report))
    result = verifier.verify(path)
    assert result["ok"] and result["verified_pairs"] == 16
    assert result["summary"] == {"baseline": 0, "tiled": 0}
    assert not result["registration_qualified"] and not result["neural_inference_replayed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_pair",
        "duplicate_pair",
        "missing_tile",
        "false_iou",
        "false_pass",
        "false_summary",
        "false_control",
        "provenance",
        "approval",
    ],
)
def test_full_verifier_rejects_modified_evidence(report_fixture, mutation):
    path, report = report_fixture
    if mutation == "missing_pair":
        report["rows"].pop()
    elif mutation == "duplicate_pair":
        report["rows"][-1] = copy.deepcopy(report["rows"][0])
    elif mutation == "missing_tile":
        report["rows"][0]["flags"][0]["records"].pop()
    elif mutation == "false_iou":
        report["rows"][0]["scores"]["tiled"]["directions"][0]["iou"] = 0.9
    elif mutation == "false_pass":
        report["rows"][0]["scores"]["tiled"]["joint_pass"] = True
    elif mutation == "false_summary":
        report["summary"]["tiled"] = 1
    elif mutation == "false_control":
        report["controls"][0]["pck3"] = 1.0
    elif mutation == "provenance":
        report["input_and_source_sha256"].pop("fixture/panel.json")
    else:
        report["registration_qualified"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verifier.verify(path)
