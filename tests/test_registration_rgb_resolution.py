import cv2
import numpy as np
import pytest

from scripts.probe_registration_rgb_resolution import (
    ignored_bytecode_paths,
    prepare_input,
    restore_to_base,
)


def test_baseline_preprocessing_parity():
    native = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    gray, transform = prepare_input(native, (360, 640), 72, 640)
    expected = cv2.cvtColor(
        cv2.cvtColor(cv2.resize(native, (640, 360)), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY
    )[72:]
    np.testing.assert_array_equal(gray, expected)
    np.testing.assert_array_equal(restore_to_base([[8, 16]], transform), [[8, 88]])


def test_highres_pixel_center_roundtrip_and_common_header():
    native = np.zeros((1080, 1920, 3), dtype=np.uint8)
    gray, transform = prepare_input(native, (360, 640), 72, 1280)
    assert gray.shape == (576, 1280)
    assert transform == dict(scale=2, top=144, full_shape=[720, 1280])
    base = np.array([[20.0, 100.0], [639.0, 359.0]])
    high = (base + 0.5) * 2 - 0.5 - [0, 144]
    np.testing.assert_allclose(restore_to_base(high, transform), base)
    # First high-resolution pixel center is a quarter base pixel before the
    # base center. Do not replace this with simple endpoint-coordinate scaling.
    np.testing.assert_allclose(restore_to_base([[0.0, 0.0]], transform), [[-0.25, 71.75]])


@pytest.mark.parametrize("width", [320, 1000, 1279])
def test_invalid_resolution_rejected(width):
    with pytest.raises(ValueError):
        prepare_input(np.zeros((1080, 1920, 3), dtype=np.uint8), (360, 640), 72, width)


def test_cache_exception_does_not_allow_vendor_source_changes():
    assert ignored_bytecode_paths("?? src/__pycache__/x.cpython-311.pyc\n") == [
        "src/__pycache__/x.cpython-311.pyc"
    ]
    for row in (" M src/x.py", "?? src/x.py", "?? src/cache.pyc", " M src/__pycache__/x.pyc"):
        with pytest.raises(ValueError, match="dirty"):
            ignored_bytecode_paths(row)
