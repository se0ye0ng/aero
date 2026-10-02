import json

import numpy as np
import pytest

from aero_ir.utils.manifest import file_sha256
from scripts import probe_ms2_intrinsic_confirmation as probe
from scripts.prepare_ms2_intrinsic_confirmation import make_plan, quarter_frames
from scripts.probe_ms2_intrinsic_confirmation import score_cameras


def candidate_report():
    names = [
        "author_calibration",
        "xoftr_640_rotation_only",
        "minima_xoftr_rotation_only",
        "xoftr_640_rotation_plus_intrinsic",
        "minima_xoftr_rotation_plus_intrinsic",
    ]
    return dict(
        candidates=[
            dict(
                name=name,
                rotation=np.eye(4).tolist(),
                intrinsic=np.diag([400.0, 400.0, 1.0]).tolist(),
            )
            for name in names
        ]
    )


def test_quarter_panel_disjoint_from_original_and_first_confirmation():
    assert quarter_frames(["000000", "000696", "001392"]) == ["000174", "000870"]
    with pytest.raises(ValueError, match="overlaps"):
        quarter_frames(["000000", "000002"])


def test_plan_freezes_complete_candidate_cameras_and_disallows_approval():
    base = dict(sequence="sequence", frame_count=1393, frame_ids=["000000", "000696", "001392"])
    plan = make_plan(base, candidate_report())
    assert len(plan["candidates"]) == 5
    assert plan["previous_frames"] == ["000000", "000696", "001392", "000348", "001044"]
    assert not plan["registration_qualified"] and not plan["generator_training_approved"]
    assert not plan["refit_allowed"] and not plan["frame_exclusion_allowed"]


def test_duplicate_candidate_rejected():
    report = candidate_report()
    report["candidates"].append(report["candidates"][0])
    with pytest.raises(ValueError, match="duplicate"):
        make_plan({}, report)


def test_invalid_or_unconverged_camera_rejected():
    report = candidate_report()
    report["candidates"][0]["rotation"][0][0] = 2.0
    with pytest.raises(ValueError, match="orthogonal"):
        make_plan({}, report)
    report = candidate_report()
    report["candidates"][-1].update(
        parameters=[0.0] * 4, optimizer_success=False, bound_hit=[False] * 4
    )
    with pytest.raises(ValueError, match="unconverged"):
        make_plan({}, report)


def test_intrinsic_candidate_keeps_reference_denominator_when_projection_leaves_image():
    k = np.eye(3)
    bad = k.copy()
    bad[0, 2] = 10000.0
    cameras = {
        name: dict(transform=np.eye(4), target_intrinsic=intrinsic)
        for name, intrinsic in [("author", k), ("bad", bad)]
    }
    xy = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    scores, supported = score_cameras(
        xy, xy, np.array([10.0, 10.0, np.nan]), k, k, np.eye(4), cameras
    )
    np.testing.assert_array_equal(supported, [True, True, False])
    assert scores["author"]["all_matches"]["fraction_all"]["3.0"] == 2 / 3
    assert scores["bad"]["fixed_supported"]["points"] == 2
    assert scores["bad"]["fixed_supported"]["fraction_all"]["3.0"] == 0


def test_changed_intrinsics_change_projection_in_native_coordinates():
    k = np.eye(3)
    changed = np.diag([2.0, 2.0, 1.0])
    cameras = {"candidate": dict(transform=np.eye(4), target_intrinsic=changed)}
    xy = np.array([[10.0, 10.0]])
    scores, _ = score_cameras(xy, 2 * xy, np.array([10.0]), k, k, np.eye(4), cameras)
    assert scores["candidate"]["fixed_supported"]["conditional_median_px"] == 0


def test_empty_camera_case_is_retained():
    camera = dict(transform=np.eye(4), target_intrinsic=np.eye(3))
    empty = np.empty((0, 2))
    scores, supported = score_cameras(
        empty, empty, np.empty(0), np.eye(3), np.eye(3), np.eye(4), {"author": camera}
    )
    assert supported.size == 0
    assert scores["author"]["fixed_supported"]["points"] == 0
    assert scores["author"]["fixed_supported"]["fraction_all"]["3.0"] is None


def synthetic_confirmation(tmp_path, monkeypatch):
    camera = dict(transform=np.eye(4).tolist(), target_intrinsic=np.eye(3).tolist())
    frames = [f"{i:06d}" for i in range(15)]
    plan = dict(frame_ids=frames, candidates={"author_calibration": camera})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(probe, "PLAN_PATH", plan_path)
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}")
    xy, depth = np.array([[1.0, 1.0]]), np.array([10.0])
    scores, support = score_cameras(
        xy, xy, depth, np.eye(3), np.eye(3), np.eye(4), plan["candidates"]
    )
    frame_rows, rows = [], []
    for index, frame in enumerate(frames):
        path = tmp_path / f"stereo_{frame}.npz"
        np.savez(path, disparity=np.ones((5, 5)), valid=np.ones((5, 5), bool))
        frame_rows.append(dict(frame=frame, stereo_file=path.name, stereo_sha256=file_sha256(path)))
        for model in ("xoftr_640", "minima_xoftr"):
            for condition in ("paired", "unrelated_thermal"):
                path = tmp_path / f"{model}_{frame}_{condition}.npz"
                np.savez(
                    path, source_xy=xy, target_xy=xy, stereo_depth_m=depth, author_supported=support
                )
                rows.append(
                    dict(
                        model=model,
                        frame=frame,
                        condition=condition,
                        target_frame=(frame if condition == "paired" else frames[(index + 7) % 15]),
                        artifact=path.name,
                        sha256=file_sha256(path),
                        scores=scores,
                    )
                )
    report = dict(
        schema="ms2_intrinsic_confirmation_v1",
        calibration_refitted=False,
        plan_sha256=file_sha256(plan_path),
        input_and_source_sha256={},
        preflight_sha256=file_sha256(preflight),
        frames=frame_rows,
        rows=rows,
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    protocol = (
        plan,
        {},
        {},
        {},
        dict(K_rgbL=np.eye(3), K_thrL=np.eye(3)),
        10.0,
        {frame: np.eye(4) for frame in frames},
        {},
    )
    return path, protocol, report


def test_complete_saved_array_verification(tmp_path, monkeypatch):
    path, protocol, _ = synthetic_confirmation(tmp_path, monkeypatch)
    result = probe.verify(path, protocol)
    assert len(result["aggregate"]) == 4
    assert len(result["verified_artifact_sha256"]) == 75
    assert all(x["equal_frame_score_missing_as_zero"] == 1 for x in result["aggregate"])
    assert not result["registration_qualified"]


@pytest.mark.parametrize(
    "change", ["score", "pair", "missing_case", "depth", "support", "artifact"]
)
def test_verifier_rejects_changed_evidence(tmp_path, monkeypatch, change):
    path, protocol, report = synthetic_confirmation(tmp_path, monkeypatch)
    row = report["rows"][0]
    if change == "score":
        row["scores"]["author_calibration"]["fixed_supported"]["fraction_all"]["3.0"] = 0.5
    elif change == "pair":
        row["target_frame"] = "000001"
    elif change == "missing_case":
        report["rows"].pop()
    else:
        artifact = tmp_path / row["artifact"]
        with np.load(artifact, allow_pickle=False) as arrays:
            data = dict(arrays)
        if change == "depth":
            data["stereo_depth_m"] = np.array([20.0])
        elif change == "support":
            data["author_supported"] = np.array([False])
        else:
            data["target_xy"] = np.array([[2.0, 2.0]])
        np.savez(artifact, **data)
        if change != "artifact":
            row["sha256"] = file_sha256(artifact)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        probe.verify(path, protocol)
