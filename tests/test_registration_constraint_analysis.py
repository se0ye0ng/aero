"""Saved-measurement attribution is not model inference or physical ground truth."""

import copy
import json

import numpy as np
import pytest

from aero_ir.registration.qualification_v4 import direction_pass, split_report
from scripts.analyze_registration_constraints import analyze, failures, run, target_scale_summary


def good():
    return {
        "finite_field": True,
        "box_in_bounds": True,
        "bbox_iou": 0.6,
        "centroid_shift_fraction": 0.25,
        "absolute_area_ratio_change": 0.5,
        "valid_fraction": 0.9,
        "cycle_valid_fraction": 0.9,
        "positive_jacobian_fraction": 0.99,
        "roi_valid_fraction": 1.0,
        "roi_positive_jacobian_fraction": 1.0,
        "cycle_p95_pixels": 1.0,
        "roi_cycle_p95_pixels": 1.0,
        "roi_cycle_max_pixels": 2.0,
        "roi_diagonal_pixels": 20.0,
    }


def pair(sequence="a", index=0):
    return {
        "sequence_id": sequence,
        "frame_index": index,
        "ir_to_rgb_points": good(),
        "rgb_to_ir_points": good(),
    }


def test_inclusive_thresholds_match_gate():
    assert direction_pass(good())
    assert failures(good()) == set()


@pytest.mark.parametrize("field", list(good()))
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "missing"])
def test_missing_nonfinite_or_invalid_measurements_fail(field, value):
    row = good()
    row[field] = value
    assert field in failures(row)
    assert not direction_pass(row)


def test_small_target_cycle_threshold_uses_diagonal():
    row = good()
    row["roi_diagonal_pixels"] = 4
    assert failures(row) == {"roi_cycle_p95_pixels", "roi_cycle_max_pixels"}


def test_both_directions_and_overlapping_failures_do_not_double_count_frames():
    first, second = pair(), pair("b")
    first["ir_to_rgb_points"]["bbox_iou"] = 0.3
    first["rgb_to_ir_points"]["bbox_iou"] = 0.4
    second["ir_to_rgb_points"]["bbox_iou"] = 0.3
    second["rgb_to_ir_points"]["cycle_p95_pixels"] = 4
    result = analyze([first, second])
    assert result["joint_failed_frame_count"] == 2
    assert result["failed_frames_by_constraint_overlapping"] == {
        "bbox_iou": 2,
        "cycle_p95_pixels": 1,
    }
    hypothetical = result["hypothetical_perfect_group_not_an_achieved_result"]
    assert hypothetical["alignment"]["joint_frame_pass_rate"] == 0.5
    assert hypothetical["cycle"]["joint_frame_pass_rate"] == 0
    assert result["minimum_frame_repairs_for_joint_threshold_only"] == 2


def test_sequence_macro_not_confused_with_frame_micro():
    rows = [pair("a", n) for n in range(3)] + [pair("b")]
    rows[-1]["ir_to_rgb_points"]["bbox_iou"] = 0.1
    result = analyze(rows)
    assert result["observed"]["joint_frame_pass_rate"] == 0.75
    assert result["observed"]["sequence_macro_pass_rate"] == 0.5
    cycle = result["hypothetical_perfect_group_not_an_achieved_result"]["cycle"]
    assert cycle["sequence_macro_pass_rate"] == 0.5


def test_empty_duplicate_and_summary_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        analyze([])
    with pytest.raises(ValueError, match="duplicate"):
        analyze([pair(), pair()])
    path = tmp_path / "screen.json"
    path.write_text(json.dumps({"rows": [pair()], "summary": {}}))
    with pytest.raises(ValueError, match="summary"):
        run(path, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_saved_screen_input_unchanged_and_output_no_overwrite(tmp_path):
    rows = [pair()]
    screen = {
        "rows": rows,
        "summary": split_report(rows),
        "evaluated_split": "train",
        "not_independent_of_training": True,
    }
    path, output = tmp_path / "screen.json", tmp_path / "out.json"
    path.write_text(json.dumps(screen))
    before = path.read_bytes()
    result = run(path, output)
    assert result["generator_training_eligible"] == "hold_not_qualified"
    assert result["input_provenance"]["not_independent_of_training"]
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        run(path, output)


def test_jsonl_keeps_splits_separate_and_does_not_assert_completion(tmp_path):
    a = pair()
    b = copy.deepcopy(a)
    a["split"], b["split"] = "train", "val"
    b["ir_to_rgb_points"]["finite_field"] = False
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    result = run(path, tmp_path / "report.json")
    assert result["splits"]["train"]["joint_failed_frame_count"] == 0
    assert result["splits"]["val"]["joint_failed_frame_count"] == 1
    assert "completion not established" in result["input_provenance"]["scope"]


def test_empty_stream_does_not_write_analysis(tmp_path):
    source, destination = tmp_path / "empty.jsonl", tmp_path / "report.json"
    source.write_text("")
    with pytest.raises(ValueError, match="empty"):
        run(source, destination)
    assert not destination.exists()


def test_target_scale_uses_both_dimensions_and_rejects_unbound_pixels(monkeypatch, tmp_path):
    from scripts import train_antiuav300_registration_v7 as trainer

    def batch(cache, selection):
        side = [3, 4, 8, 16][selection[0][1]]
        box = np.array([[0.5, 0.5, side / 256, side / 128]], dtype=np.float32)
        target = box.copy()
        target[:, 2:] *= 2
        return {
            "source_boxes": box,
            "target_boxes": target,
            "visible": np.zeros((1, 128, 256, 3), np.uint8),
            "infrared": np.zeros((1, 128, 256, 3), np.uint8),
        }, "a" * 64

    monkeypatch.setattr(trainer, "read_batch", batch)
    rows = [pair(str(i), i) for i in range(4)]
    samples = [{"selection": [[str(i), i]], "array_sha256": "a" * 64} for i in range(4)]
    summary = target_scale_summary(rows, samples, tmp_path)
    assert all(v["frames"] == v["passed"] == 1 for v in summary["strata"].values())
    assert summary["observations"][1]["short_sides_visible_ir_pixels"] == [4.0, 8.0]
    samples[0]["array_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="observations differ"):
        target_scale_summary(rows, samples, tmp_path)
    with pytest.raises(ValueError, match="one saved sample"):
        target_scale_summary(rows, samples[:1], tmp_path)
    samples[0]["selection"] = [["wrong_sequence", 0]]
    with pytest.raises(ValueError, match="selection differs"):
        target_scale_summary(rows, samples, tmp_path)
