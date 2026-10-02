import copy
import sys

import numpy as np
import pytest

from aero_ir.registration.raft_stereo import (
    COMMIT,
    ITERATIONS,
    WEIGHTS_SHA,
    consistency,
    rectify_images,
)
from aero_ir.registration.temporal_stereo import rectification
from scripts import analyze_ms2_raft_stereo as module


def fixture_protocol():
    case = dict(
        sensor="thr",
        frame="000696",
        condition="paired_timed",
        right_frame="000696",
        geometry_available=True,
    )
    baseline = dict(rows=[case])
    result = dict(
        schema="ms2_raft_stereo_v1",
        rows=[case.copy()],
        registration_qualified=False,
        generator_training_approved=False,
    )
    preflight = dict(
        schema="ms2_raft_stereo_plan_v1",
        vendor_commit=COMMIT,
        checkpoint_sha256=WEIGHTS_SHA,
        iterations=ITERATIONS,
        precision="float32",
        correlation="reg",
        image_resize=False,
        padding_divisor=32,
        equivalent_disparity_focal_px=200.0,
        positive_disparity_range=[0.0, 128.0],
        lr_tolerance_px=1.0,
        device="cuda",
        smoke_only=False,
        no_finetuning=True,
        registration_qualified=False,
        generator_training_approved=False,
        input="native RGB color; thermal original window replicated to 3 channels",
        representation_matches_sgbm=False,
        support="same 5x5 remapping boundary and four-neighbor sampling rules",
        gpu="test fixture, not a real result",
        cases=[case.copy()],
    )
    return baseline, result, preflight


def test_protocol_accepts_exact_case_and_rejects_changed_pairing():
    baseline, result, pre = fixture_protocol()
    module.validate_protocol(baseline, result, pre, 200.0)
    result["rows"][0]["right_frame"] = "000000"
    with pytest.raises(ValueError, match="pairing"):
        module.validate_protocol(baseline, result, pre, 200.0)


@pytest.mark.parametrize(
    "change",
    [
        dict(iterations=16),
        dict(image_resize=True),
        dict(device="cpu"),
        dict(equivalent_disparity_focal_px=201.0),
        dict(smoke_only=True),
        dict(registration_qualified=True),
    ],
)
def test_changed_inference_protocol_is_rejected(change):
    baseline, result, pre = fixture_protocol()
    pre.update(change)
    with pytest.raises(ValueError, match="protocol"):
        module.validate_protocol(baseline, result, pre, 200.0)


def test_duplicate_case_and_qualification_claim_are_rejected():
    baseline, result, pre = fixture_protocol()
    duplicate = copy.deepcopy(result)
    duplicate["rows"].append(duplicate["rows"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        module.validate_protocol(baseline, duplicate, pre, 200.0)
    result["generator_training_approved"] = True
    with pytest.raises(ValueError, match="qualification"):
        module.validate_protocol(baseline, result, pre, 200.0)


def test_validity_is_rebuilt_not_trusted_from_saved_mask():
    k = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 48.0], [0.0, 0.0, 1.0]])
    b = np.eye(4)
    b[0, 3] = -0.3
    geometry = rectification(k, k, b, (96, 320))
    blank = np.zeros((96, 320), np.uint8)
    _, supports = rectify_images(blank, blank, geometry)
    d = np.full((96, 320), 4.0)
    arrays = consistency(d, d, *supports)
    module.validate_support(arrays, geometry)
    arrays["valid"][40, 170] = not arrays["valid"][40, 170]
    with pytest.raises(ValueError, match="support"):
        module.validate_support(arrays, geometry)


def test_unavailable_pose_cannot_contain_estimated_stereo_map():
    arrays = dict(
        estimated_depth_m=np.array([np.nan]), equivalent_disparity_errors=np.array([np.nan])
    )
    module.validate_support(arrays, {})
    arrays["disparity"] = np.ones((8, 8))
    with pytest.raises(ValueError, match="unexpected"):
        module.validate_support(arrays, {})


def test_extra_coverage_is_not_confused_with_common_point_accuracy():
    stats = module.paired_statistics(
        np.array([0.2, np.nan]), np.array([0.2, 2.0]), np.array([10.0, 10.0])
    )
    row = dict(sensor="thr", condition="paired_timed", strata=stats)
    result = module.aggregate_comparison([row])[0]
    assert result["baseline"]["supported"] == 1 and result["raft"]["supported"] == 2
    assert result["baseline"]["equal_frame_full_fraction_within_px"]["3.0"] == 0.5
    assert result["raft"]["equal_frame_full_fraction_within_px"]["3.0"] == 1.0
    assert result["common_frame_medians"] == dict(improved=0, worsened=0, tied=1)


def test_unavailable_frame_remains_failure_in_full_reference_mean():
    depth = np.array([10.0, 10.0])
    rows = [
        dict(
            sensor="thr",
            condition="paired_timed",
            strata=module.paired_statistics(np.zeros(2), np.zeros(2), depth),
        ),
        dict(
            sensor="thr",
            condition="paired_timed",
            strata=module.paired_statistics(np.full(2, np.nan), np.full(2, np.nan), depth),
        ),
    ]
    result = module.aggregate_comparison(rows)[0]
    assert result["planned_frames"] == 2 and result["reference_points"] == 4
    assert result["raft"]["equal_frame_full_fraction_within_px"]["1.0"] == 0.5
    assert result["common_frames"] == 1


def test_missing_gpu_report_fails_before_scanning_assets_or_writing_output(tmp_path, monkeypatch):
    out = tmp_path / "no_analysis.json"
    monkeypatch.setattr(
        sys, "argv", ["analyze", "--run-dir", str(tmp_path / "absent"), "--out", str(out)]
    )

    def forbidden():
        raise AssertionError("should not scan model assets without a GPU report")

    monkeypatch.setattr(module, "dependencies", forbidden)
    with pytest.raises(FileNotFoundError, match="GPU report not found"):
        module.main()
    assert not out.exists()
