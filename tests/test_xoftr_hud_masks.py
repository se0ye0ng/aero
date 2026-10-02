import json

import cv2
import numpy as np
import pytest

from aero_ir.utils.manifest import file_sha256
from scripts.probe_xoftr_hud_masks import (
    fill_image,
    filter_matches,
    load_masks,
    merged_and_excluded,
    retained_points,
    roi_support,
)


def test_fill_uses_only_valid_pixels_without_mutating_image():
    image = np.array([[200, 200], [1, 9]], dtype=np.uint8)
    mask = np.array([[True, True], [False, False]])
    before = image.copy()
    for method in ("median", "mean"):
        filled, value = fill_image(image, mask, method)
        assert value == 5
        assert filled.tolist() == [[5, 5], [1, 9]]
    assert np.array_equal(image, before)


def test_fill_sensitivity_different_statistics():
    image = np.array([[200, 1, 2, 30]], dtype=np.uint8)
    mask = np.array([[True, False, False, False]])
    assert fill_image(image, mask, "median")[1] == 2
    assert fill_image(image, mask, "mean")[1] == 11


def test_dilation_header_and_input_immutability():
    mask = np.zeros((64, 64), dtype=bool)
    mask[32, 32] = True
    merged, excluded = merged_and_excluded(mask, 4)
    assert mask.sum() == 1
    assert merged[:4].all() and not merged[4].any()
    assert excluded[24:41, 24:41].all()
    assert not excluded[23, 32] and not excluded[41, 32]
    assert excluded[:12].all()
    assert not excluded[12, 0]


def test_exclude_both_endpoints_and_reverse_direction():
    mask = np.zeros((20, 20), dtype=bool)
    mask[4, 4] = True
    points = np.array([[2, 2], [4, 4], [18, 18]], dtype=float)
    paired = points[::-1].copy()
    arrays = (points, paired, np.ones(3), paired, points, np.ones(3))
    kept = filter_matches(arrays, [mask, mask])
    assert len(kept[0]) == len(kept[3]) == 2
    assert not retained_points(np.array([[np.nan, 2], [-1, 4], [20, 3]]), mask).any()


def test_roi_coverage_does_not_change_mask():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 2] = True
    result = roi_support([1, 1, 3, 3], mask, mask)
    assert result["roi_pixel_centres"] == 9
    assert result["target_occluded_fraction"] == pytest.approx(1 / 9)


def test_all_masked_and_dilated_empty_support_reject():
    mask = np.ones((20, 20), dtype=bool)
    with pytest.raises(ValueError, match="no valid"):
        fill_image(np.ones((20, 20)), mask, "mean")
    with pytest.raises(ValueError, match="no valid"):
        merged_and_excluded(mask, 2)
    mask[10, 10] = False
    with pytest.raises(ValueError, match="no evaluation"):
        merged_and_excluded(mask, 2)


def fixture_manifest(tmp_path, value=255):
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[20, 20] = value
    image_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), mask)
    inputs = {
        m: {"resized_shape": [40, 40], "resized_rgb_sha256": "imagehash"}
        for m in ("visible", "infrared")
    }
    baseline = {"inputs": [{"sequence_id": "seq", "frame_index": 1, "inputs": inputs}]}
    entry = {
        "path": "mask.png",
        "sha256": file_sha256(image_path),
        "shape": [40, 40],
        "image_sha256": "imagehash",
    }
    manifest = {
        "schema": "aero_hud_masks_v1",
        "reviewed_by": "human one",
        "model_blind_attestation": True,
        "baseline_report_sha256": "baselinehash",
        "pairs": [
            {
                "sequence_id": "seq",
                "frame_index": 1,
                "masks": {"visible": dict(entry), "infrared": dict(entry)},
            }
        ],
    }
    return baseline, manifest


def test_valid_reviewed_binary_mask_loads(tmp_path):
    baseline, manifest = fixture_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    _, masks = load_masks(path, baseline, "baselinehash")
    assert masks[("seq", 1)][0].dtype == np.bool_
    assert masks[("seq", 1)][0].sum() == 1


@pytest.mark.parametrize(
    "failure", ["reviewer", "attestation", "baseline", "image", "hash", "missing"]
)
def test_unreviewed_and_provenance_fail_closed(tmp_path, failure):
    baseline, manifest = fixture_manifest(tmp_path)
    if failure == "reviewer":
        manifest["reviewed_by"] = " "
    elif failure == "attestation":
        manifest["model_blind_attestation"] = False
    elif failure == "baseline":
        manifest["baseline_report_sha256"] = "wrong"
    elif failure == "image":
        manifest["pairs"][0]["masks"]["visible"]["image_sha256"] = "wrong"
    elif failure == "hash":
        manifest["pairs"][0]["masks"]["visible"]["sha256"] = "wrong"
    else:
        manifest["pairs"] = []
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_masks(path, baseline, "baselinehash")


def test_nonbinary_mask_rejected(tmp_path):
    baseline, manifest = fixture_manifest(tmp_path, value=127)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binary"):
        load_masks(path, baseline, "baselinehash")
