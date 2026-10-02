import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aero_ir.registration.roi_geometry import (
    centred_transform,
    legacy_transform,
    prepare_centred,
    restore_cached,
    transform_points,
)
from aero_ir.registration.silhouette_search import matrix_for, score, search


@pytest.mark.parametrize("box", ([0, 0, 20, 10], [80, 85, 99, 99], [20, 30, 70, 50]))
def test_centre_and_aspect_ratio(box):
    box = np.array(box, float)
    h = centred_transform(box)
    assert h[0, 0] == h[1, 1]
    assert np.allclose(transform_points([(box[:2] + box[2:]) / 2], h), [[127.5, 127.5]])
    assert np.allclose(
        transform_points(transform_points(box.reshape(2, 2), h), np.linalg.inv(h)),
        box.reshape(2, 2),
    )


def test_padding_never_valid():
    image = np.full((100, 100), 100, np.uint8)
    data = prepare_centred(image, [0, 0, 20, 10], np.zeros_like(image, bool))
    assert not data["valid"][0, 0]
    assert data["excluded"][0, 0]
    assert data["valid"][128, 128]


def test_cached_border_restoration():
    meta = {"bounds_xyxy": [1497, 877, 1827, 1080], "scale_xy": [256 / 330, 256 / 203]}
    old = legacy_transform(meta)
    native_box = np.array([1607, 1019, 1717, 1065])
    cb = transform_points(native_box.reshape(2, 2), old).ravel()
    data = restore_cached(
        np.zeros((3, 256, 256), bool),
        np.zeros((256, 256), np.uint8),
        np.zeros((256, 256), bool),
        cb,
        meta,
    )
    assert np.allclose((data["box"][:2] + data["box"][2:]) / 2, [127.5, 127.5])
    assert np.allclose(data["meta"]["native_box"], native_box)
    h = np.array(data["meta"]["native_to_crop"])
    assert h[0, 0] == h[1, 1]
    assert not data["valid"][-1].any()


def test_padding_overlap_rejected():
    mask = np.zeros((64, 64), bool)
    mask[20:30, 20:30] = True
    valid = np.ones_like(mask)
    valid[25, 25] = False
    assert score(mask, mask, np.eye(3), valid, np.ones_like(mask)) is None


def test_known_shift_search_and_inverse():
    mask = np.zeros((256, 256), bool)
    mask[100:120, 100:150] = True
    mask[120:150, 100:115] = True
    target = np.roll(mask, 16, axis=1)
    valid = np.ones_like(mask)
    r = search(mask, target, valid, valid)
    assert r["status"] == "fit_pseudo_masks_only"
    assert r["selected"]["dice_loss"] < 0.01
    assert abs(r["parameters"]["dx"] - 16) <= 1
    assert np.allclose(np.array(r["matrix"]) @ r["inverse_matrix"], np.eye(3))
    assert r["determinant"] > 0


def test_similarity_has_positive_jacobian():
    assert np.linalg.det(matrix_for([0.6, 30, -32, 16], 256)) == pytest.approx(0.36)


def test_empty_mask_search_abstains():
    a = np.zeros((256, 256), bool)
    v = np.ones_like(a)
    assert search(a, a, v, v)["status"] == "no_foreground_safe_transform"


def test_wrapper_refuses_existing_directory_before_gpu(tmp_path):
    env = dict(os.environ, AERO_SAM_V2_OUTPUT=str(tmp_path))
    result = subprocess.run(
        ["bash", "scripts/run_registration_sam_v2.sh"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "Refusing to overwrite" in result.stderr


def test_wrapper_rejects_multiline_path_before_gpu():
    env = dict(os.environ, AERO_SAM_V2_OUTPUT="/tmp/invalid\noutput")
    result = subprocess.run(
        ["bash", "scripts/run_registration_sam_v2.sh"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "Multiline path rejected" in result.stderr


def test_native_transform_composition():
    rgb = centred_transform([1200, 850, 1280, 900])
    ir = centred_transform([400, 370, 432, 390])
    h = matrix_for([1.1, 8, 3, -2], 256)
    native = np.linalg.inv(ir) @ h @ rgb
    points = np.array([[1210.0, 865.0], [1250.0, 880.0]])
    expected = transform_points(
        transform_points(transform_points(points, rgb), h), np.linalg.inv(ir)
    )
    assert np.allclose(transform_points(points, native), expected)
    assert np.allclose(
        transform_points(transform_points(points, native), np.linalg.inv(native)), points
    )


def test_old_sam_report_and_sources_remain_unchanged():
    # Local artifact integration check; CI need not possess user experiment artifacts.
    import json

    from aero_ir.utils.manifest import file_sha256

    path = Path("experiments/registration_sam_train16_v1/report.json")
    if not path.exists():
        pytest.skip("local v1 artifacts unavailable")
    report = json.loads(path.read_text())
    assert all(file_sha256(name) == digest for name, digest in report["sources"].items())
