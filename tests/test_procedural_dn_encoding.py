import numpy as np
import pytest

from aero_ir.data.preprocess import _canonical_hash, apply_uint16_linear_preprocess
from aero_ir.generate.dn_encoding import encode_procedural, procedural_encoding
from aero_ir.generate.synthetic_baseline import SyntheticBaselineGenerator


def window():
    s = dict(
        schema_version=1,
        fit_split="train",
        source_manifest_sha256="fixture",
        lower_dn=6076,
        upper_dn=8097,
        output_max=255.0,
    )
    s["preprocess_sha256"] = _canonical_hash(s)
    return s


def test_default_raw_output_would_be_black_but_encoding_preserves_range():
    p = window()
    spec = procedural_encoding(SyntheticBaselineGenerator(), p)
    image = np.array([[3480.0, 3564.0, 3648.0]])
    assert not apply_uint16_linear_preprocess(image.astype(np.uint16), p).any()
    encoded = encode_procedural(image, spec, p)
    assert encoded.shape == image.shape and encoded.dtype == np.uint16
    actual = apply_uint16_linear_preprocess(encoded, p)[0, :, 0]
    np.testing.assert_allclose(actual, [0, 127.5, 255], atol=0.07)


def test_no_image_specific_contrast_stretching():
    p = window()
    spec = procedural_encoding(SyntheticBaselineGenerator(), p)
    x = np.array([[3500.0, 3550.0]])
    y = np.array([[3480.0, 3500.0, 3550.0, 3648.0]])
    np.testing.assert_array_equal(
        encode_procedural(x, spec, p)[0], encode_procedural(y, spec, p)[0, 1:3]
    )


def test_out_of_range_and_changed_contract_fail():
    p = window()
    spec = procedural_encoding(SyntheticBaselineGenerator(), p)
    with pytest.raises(ValueError, match="outside declared"):
        encode_procedural(np.array([[10000.0]]), spec, p)
    changed = {**spec, "source_range": [0, 65535]}
    with pytest.raises(ValueError, match="hash mismatch"):
        encode_procedural(np.array([[3500.0]]), changed, p)


def test_nontraining_window_and_postsensor_output_rejected():
    p = window()
    p["fit_split"] = "val"
    p["preprocess_sha256"] = _canonical_hash(
        {k: v for k, v in p.items() if k != "preprocess_sha256"}
    )
    with pytest.raises(ValueError, match="training"):
        procedural_encoding(SyntheticBaselineGenerator(), p)
    with pytest.raises(ValueError, match="before a sensor"):
        procedural_encoding(SyntheticBaselineGenerator(sensor=object()), window())
