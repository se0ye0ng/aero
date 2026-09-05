"""The sensor chain must preserve shape, stay differentiable, and actually do something."""

import pytest

torch = pytest.importorskip("torch")

from aero_ir.sensor import (  # noqa: E402
    AGCQuantise,
    FixedPatternNoise,
    IRSensorPipeline,
    MTFBlur,
    NETDNoise,
)


def _x(b=2, c=1, h=64, w=64):
    return torch.rand(b, c, h, w) * 100.0


def test_pipeline_preserves_shape():
    pipe = IRSensorPipeline(
        [
            MTFBlur(0.35),
            NETDNoise(50.0, 12.0),
            FixedPatternNoise(),
            AGCQuantise(),
        ]
    )
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


def test_lazy_fixed_pattern_noise_survives_checkpoint_reload():
    x = _x(b=1, h=32, w=48)
    original = FixedPatternNoise()
    expected = original(x)
    restored = FixedPatternNoise()
    restored.load_state_dict(original.state_dict())
    assert torch.equal(restored(x), expected)


def test_pipeline_applies_to_numpy_images():
    import numpy as np

    pipeline = IRSensorPipeline([AGCQuantise(mode="minmax")])
    outputs = pipeline.apply_numpy([np.arange(64, dtype=np.float32).reshape(8, 8)])
    assert len(outputs) == 1
    assert outputs[0].shape == (8, 8)
    assert 0.0 <= float(outputs[0].min()) <= float(outputs[0].max()) <= 1.0
