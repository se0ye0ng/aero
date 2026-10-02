import json

import numpy as np
import pytest

import scripts.verify_minima_roma as verifier
from aero_ir.utils.manifest import file_sha256
from scripts.probe_minima_roma_gpu import evaluate


@pytest.fixture
def synthetic_report(tmp_path, monkeypatch):
    x, y = np.meshgrid(2 * (np.arange(8) + 0.5) / 8 - 1, 2 * (np.arange(8) + 0.5) / 8 - 1)
    g = np.stack([x, y], axis=-1)
    warp = np.concatenate([np.concatenate([g, g], -1)] * 2, axis=1)
    certainty = np.ones((8, 16))
    artifact = tmp_path / "identity.npz"
    np.savez_compressed(artifact, warp=warp, certainty=certainty)
    cases = [
        dict(
            sequence_id=f"synthetic{i}",
            frame_index=10,
            header_rows=[0, 0],
            boxes_xyxy=[[20, 20, 40, 40]] * 2,
        )
        for i in range(16)
    ]
    metadata = {
        r["sequence_id"]: {m: dict(resized_shape=[64, 64]) for m in ("visible", "infrared")}
        for r in cases
    }
    monkeypatch.setattr(verifier, "protocol", lambda: ({}, cases, metadata))
    rows = []
    for case in cases:
        score = evaluate(
            warp, certainty, [np.empty((64, 64, 3))] * 2, case["header_rows"], case["boxes_xyxy"]
        )
        rows.append(
            dict(
                **case,
                scores=score,
                crop_shapes=[[64, 64]] * 2,
                artifact=artifact.name,
                sha256=file_sha256(artifact),
            )
        )
    shifted = warp.copy()
    shifted[:, :8, 2:] += [0.25, -0.25]
    shift_path = tmp_path / "shift.npz"
    np.savez_compressed(shift_path, warp=shifted, certainty=certainty)
    controls = [
        dict(
            kind="known_shift",
            artifact=shift_path.name,
            sha256=file_sha256(shift_path),
            queries=4,
            supported=4,
            pck3_all=1.0,
            certainty_ge_half_fraction=[1.0, 1.0],
        ),
        dict(
            kind="unrelated",
            artifact=artifact.name,
            sha256=file_sha256(artifact),
            certainty_ge_half_fraction=[1.0, 1.0],
        ),
    ]
    report = dict(
        rows=rows,
        pairs=16,
        joint_passes=16,
        controls=controls,
        input_and_source_sha256={},
        registration_qualified=False,
        generator_training_approved=False,
    )
    return tmp_path / "report.json", report


def test_reconstruction_does_not_authorize_qualification(synthetic_report):
    path, report = synthetic_report
    path.write_text(json.dumps(report))
    result = verifier.verify(path)
    assert result["ok"] and result["joint_passes"] == 16
    assert not result["registration_qualified"] and not result["neural_inference_replayed"]


@pytest.mark.parametrize("change", ["pair", "score", "summary", "header", "control", "approval"])
def test_inconsistent_report_rejected(synthetic_report, change):
    path, report = synthetic_report
    if change == "pair":
        report["rows"].pop()
    elif change == "score":
        report["rows"][0]["scores"]["iou"][0] = 0.9
    elif change == "summary":
        report["joint_passes"] = 15
    elif change == "header":
        report["rows"][0]["header_rows"] = [1, 0]
    elif change == "control":
        report["controls"][0]["pck3_all"] = 0.5
    else:
        report["registration_qualified"] = True
    path.write_text(json.dumps(report))
    with pytest.raises((ValueError, AssertionError)):
        verifier.verify(path)
