"""The sensor chain must preserve shape, stay differentiable, and actually do something."""

import pytest

torch = pytest.importorskip("torch")

from aero_ir.sensor import AGCQuantise, FixedPatternNoise, IRSensorPipeline, MTFBlur, NETDNoise


def _x(b=2, c=1, h=64, w=64):
    return torch.rand(b, c, h, w) * 100.0


def test_pipeline_preserves_shape():
    pipe = IRSensorPipeline([
        MTFBlur(0.35), NETDNoise(50.0, 12.0), FixedPatternNoise(), AGCQuantise(),
    ])
    x = _x()
    assert pipe(x).shape == x.shape


def test_empty_pipeline_is_identity():
    x = _x()
    assert torch.equal(IRSensorPipeline([])(x), x)


def test_blur_reduces_local_variance():
    x = _x()
    assert MTFBlur(0.2)(x).var() < x.var()


def test_agc_output_is_bounded_and_quantised():
    y = AGCQuantise(bit_depth=8)(_x())
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert torch.allclose(y * 255.0, torch.round(y * 255.0), atol=1e-4)


def test_chain_is_differentiable_end_to_end():
    """N2 depends on this: a detection loss must reach the generator through the sensor."""
    x = _x().requires_grad_(True)
    pipe = IRSensorPipeline([MTFBlur(0.35), NETDNoise(50.0, 12.0), AGCQuantise()])
    pipe(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0


def test_fixed_pattern_noise_is_fixed():
    fpn = FixedPatternNoise()
    x = _x(b=1)
    assert torch.equal(fpn(x) - x, fpn(x) - x)
