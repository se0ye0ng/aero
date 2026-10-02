import json

import numpy as np
import pytest
from PIL import Image

from aero_ir.utils.manifest import file_sha256
from scripts.audit_ms2_screen import inspect, verified_files


@pytest.fixture
def screen(tmp_path, monkeypatch):
    sequence = "_2021-08-06-10-59-33"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            dict(
                schema="ms2_train_screen_plan_v1",
                sequence=sequence,
                frame_count=2,
                frame_ids=["000000", "000001"],
            )
        )
    )
    monkeypatch.setattr(
        "scripts.audit_ms2_screen.audit", lambda _: {"first_training_sequence": sequence}
    )
    reports = []
    for kind in ("sync_data", "proj_depth"):
        base = tmp_path / kind / sequence
        base.mkdir(parents=True)
        for sensor in ("rgb", "thr"):
            folders = ("img_left", "img_right") if kind == "sync_data" else ("depth",)
            for folder in folders:
                target = base / sensor / folder
                target.mkdir(parents=True)
                for frame in ("000000", "000001"):
                    rgb = sensor == "rgb" and kind == "sync_data"
                    pixels = np.ones(
                        (4, 6, 3) if rgb else (4, 6), dtype=np.uint8 if rgb else np.uint16
                    )
                    Image.fromarray(pixels).save(target / f"{frame}.png")
            if kind == "sync_data":
                (base / sensor / "img_left_timestamp.txt").write_text(
                    "1628215174691250698\n1628215174791250698\n"
                )
        if kind == "sync_data":
            np.save(
                base / "calib.npy",
                {
                    "K_rgbL": np.eye(3),
                    "K_thrL": np.eye(3),
                    "R_nir2rgb": np.eye(3),
                    "R_nir2thr": np.eye(3),
                    "T_nir2rgb": np.zeros((3, 1)),
                    "T_nir2thr": np.zeros((3, 1)),
                },
            )
        files = [
            dict(path=str(p.relative_to(tmp_path)), bytes=p.stat().st_size, sha256=file_sha256(p))
            for p in base.rglob("*")
            if p.is_file()
        ]
        report = tmp_path / f"{kind}.json"
        report.write_text(
            json.dumps(
                dict(
                    schema="ms2_selected_extraction_v1",
                    kind=kind,
                    sequence=sequence,
                    complete=True,
                    required_missing=[],
                    plan_sha256=file_sha256(plan_path),
                    files=files,
                )
            )
        )
        reports.append(report)
    return tmp_path, plan_path, reports


def test_native_screen_observations_are_not_qualification(screen):
    root, plan, reports = screen
    result = inspect(root, root, *reports, plan, root)
    assert len(result["images"]) == 12
    assert result["calibration_numerical_errors"] == {}
    assert result["calibration_required_missing"] == []
    assert result["stereo_depth_dimension_errors"] == []
    assert len(result["depth_product_consistency"]) == 4
    assert result["depth_consistency_skipped_invalid_calibration"] is False
    assert result["same_index_timestamps"]["absolute_skew"]["max_ms"] == 0
    assert result["selected_thermal_minus_rgb_ns"] == {"000000": 0, "000001": 0}
    assert len(result["additional_timestamp_files_missing"]) == 5
    assert result["registration_qualified"] is False
    assert result["generator_training_approved"] is False
    json.dumps(result, allow_nan=False)


def test_changed_asset_rejected(screen):
    root, plan, reports = screen
    target = next(root.rglob("*.png"))
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        inspect(root, root, *reports, plan, root)


def test_incomplete_report_rejected(screen):
    root, plan, reports = screen
    data = json.loads(reports[0].read_text())
    data["complete"] = False
    reports[0].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        verified_files(root, reports[0], plan, "sync_data")


def test_dimension_mismatch_is_reported_without_resizing(screen):
    root, plan, reports = screen
    data = json.loads(reports[1].read_text())
    item = data["files"][0]
    path = root / item["path"]
    Image.fromarray(np.ones((5, 6), dtype=np.uint16)).save(path)
    item.update(bytes=path.stat().st_size, sha256=file_sha256(path))
    reports[1].write_text(json.dumps(data))
    result = inspect(root, root, *reports, plan, root)
    assert len(result["stereo_depth_dimension_errors"]) == 1
    assert result["stereo_depth_dimension_errors"][0]["shapes"]["depth"] == [5, 6]
    assert result["registration_qualified"] is False
