"""ONNX export and INT8 quantisation, with a latency-versus-mAP curve.

The curve is the deliverable, not the single quantised number: what matters operationally is
where accuracy starts to fall off, and that is only visible as a trade-off.
"""

from __future__ import annotations


def export_onnx(model, input_size, out_path: str, opset: int = 17) -> str:
    raise NotImplementedError


def quantize_int8(onnx_path: str, calibration_set, out_path: str) -> str:
    """Post-training static quantisation. Calibration uses the real training split."""
    raise NotImplementedError


def benchmark(engine_path: str, input_size, warmup: int = 50, iters: int = 500) -> dict:
    """Return latency percentiles and throughput on a fixed device profile."""
    raise NotImplementedError
