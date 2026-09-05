import pytest

from aero_ir.detect.yolox_adapter import YOLOX_VERSION, YOLOXDetector, yolox_backend_status


def test_yolox_detector_validates_the_protocol_pin():
    detector = YOLOXDetector(arch="yolox-s", input_size=(640, 640), num_classes=6)
    assert detector.backend_version == YOLOX_VERSION

    with pytest.raises(ValueError, match="pinned"):
        YOLOXDetector(backend_version="main")
    with pytest.raises(ValueError, match="architecture"):
        YOLOXDetector(arch="other")


def test_yolox_backend_preflight_is_explicit():
    status = yolox_backend_status()
    assert status["required_version"] == "0.3.0"
    assert isinstance(status["available"], bool)
    assert status["reason"]
