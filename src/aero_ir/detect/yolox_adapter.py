"""YOLOX training and inference behind a stable interface."""

from __future__ import annotations

from importlib import import_module, metadata
from pathlib import Path

YOLOX_VERSION = "0.3.0"
YOLOX_SOURCE_SHA256 = "1b8ee68846434408c40074f0806259fe3b9387181e562dbc435ea666533d8d08"
SUPPORTED_ARCHITECTURES = {"yolox-tiny", "yolox-s", "yolox-m", "yolox-l"}


def yolox_backend_status() -> dict[str, str | bool | int]:
    """Report whether the pinned upstream package is importable without touching a GPU."""
    try:
        installed_version = metadata.version("yolox")
    except metadata.PackageNotFoundError:
        return {
            "available": False,
            "required_version": YOLOX_VERSION,
            "installed_version": "missing",
            "reason": "install the hash-pinned requirements/yolox.txt artifact",
        }
    if installed_version != YOLOX_VERSION:
        return {
            "available": False,
            "required_version": YOLOX_VERSION,
            "installed_version": installed_version,
            "reason": "installed YOLOX version does not match the experiment pin",
        }
    try:
        import_module("yolox")
        get_exp = import_module("yolox.exp").get_exp
        experiment = get_exp(None, "yolox-s")
        experiment.num_classes = 1
        model = experiment.get_model()
    except Exception as error:
        return {
            "available": False,
            "required_version": YOLOX_VERSION,
            "installed_version": installed_version,
            "reason": f"pinned YOLOX import failed: {type(error).__name__}: {error}",
        }
    return {
        "available": True,
        "required_version": YOLOX_VERSION,
        "installed_version": installed_version,
        "reason": "pinned backend import passed",
        "cpu_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def require_yolox_backend() -> None:
    status = yolox_backend_status()
    if not status["available"]:
        raise RuntimeError(str(status["reason"]))


class YOLOXDetector:
    def __init__(
        self,
        arch: str = "yolox-s",
        input_size=(640, 640),
        num_classes: int = 80,
        pretrained: bool = False,
        weights: str | None = None,
        backend_version: str = YOLOX_VERSION,
    ) -> None:
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"unsupported YOLOX architecture: {arch}")
        if len(input_size) != 2 or any(int(value) <= 0 for value in input_size):
            raise ValueError("input_size must contain two positive integers")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if backend_version != YOLOX_VERSION:
            raise ValueError(
                f"YOLOX backend must be pinned to {YOLOX_VERSION}, got {backend_version}"
            )
        self.arch = arch
        self.input_size = tuple(input_size)
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.weights = weights
        self.backend_version = backend_version

    def preflight(self) -> dict[str, str | bool | int]:
        """Validate the upstream code pin and any requested checkpoint without CUDA."""
        status = yolox_backend_status()
        weights_status = "not requested"
        if self.weights is not None:
            weights_path = Path(self.weights)
            weights_status = "present" if weights_path.is_file() else "missing"
            if not weights_path.is_file():
                status = {
                    **status,
                    "available": False,
                    "reason": f"YOLOX weights do not exist: {weights_path}",
                }
        return {**status, "weights": weights_status}

    def fit(self, train_set, val_set, train_cfg):
        """Train only through the forthcoming manifest-aware FLIR experiment adapter."""
        require_yolox_backend()
        raise RuntimeError(
            "YOLOX is pinned, but training remains disabled until the analytics16 preprocessing "
            "and manifest-aware upstream dataset adapter are frozen"
        )

    def predict(self, images):
        require_yolox_backend()
        raise RuntimeError(
            "YOLOX inference remains disabled until preprocessing parity with training is frozen"
        )
