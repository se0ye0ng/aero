"""Explicit procedural-output encoding into a frozen train-only detector DN window.

This is numerical range encoding, not sensor or physical temperature calibration.
It leaves spatial coordinates unchanged and must be recorded as part of an arm.
"""

import numpy as np

from aero_ir.data.preprocess import verify_preprocess_spec
from aero_ir.utils.manifest import canonical_hash


def procedural_encoding(generator, preprocess):
    verify_preprocess_spec(preprocess)
    if preprocess.get("fit_split") != "train":
        raise ValueError("DN window must have been fitted on training data")
    if generator.sensor is not None:
        raise ValueError("encoding expects procedural output before a sensor pipeline")
    parameters = dict(
        base=generator.base_temp_k,
        contrast=generator.scene_contrast_k,
        response=generator.scene_response_dn_per_K,
    )
    deltas = list(generator.class_delta_k.values())
    if not deltas or not np.isfinite([*parameters.values(), *deltas]).all():
        raise ValueError("finite procedural parameters required")
    if parameters["contrast"] < 0 or parameters["response"] <= 0:
        raise ValueError("nonnegative contrast and positive response required")
    low = (parameters["base"] + min(0, min(deltas))) * parameters["response"]
    high = (parameters["base"] + parameters["contrast"] + max(0, max(deltas))) * parameters[
        "response"
    ]
    target = [preprocess["lower_dn"], preprocess["upper_dn"]]
    if low >= high or not 0 <= target[0] < target[1] <= 65535:
        raise ValueError("nondegenerate source and uint16 destination windows required")
    specification = dict(
        schema_version=1,
        kind="procedural_range_to_train_dn_window_not_physical_calibration",
        source_range=[float(low), float(high)],
        target_range=target,
        preprocess_sha256=preprocess["preprocess_sha256"],
        source_manifest_sha256=preprocess["source_manifest_sha256"],
        fit_split="train",
        quantization="round_to_nearest_even_uint16",
        class_delta_k=dict(generator.class_delta_k),
        parameters=parameters,
    )
    specification["encoding_sha256"] = canonical_hash(specification)
    return specification


def encode_procedural(image, specification, preprocess):
    """Fail on incompatible units instead of silently mapping out-of-range images."""
    unsigned = {k: v for k, v in specification.items() if k != "encoding_sha256"}
    if canonical_hash(unsigned) != specification.get("encoding_sha256"):
        raise ValueError("encoding specification hash mismatch")
    verify_preprocess_spec(preprocess)
    if specification.get("preprocess_sha256") != preprocess["preprocess_sha256"]:
        raise ValueError("encoding and detector preprocessing differ")
    if specification.get("kind") != "procedural_range_to_train_dn_window_not_physical_calibration":
        raise ValueError("unsupported encoding")
    a = np.asarray(image, dtype=float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("finite nonempty 2D procedural image required")
    low, high = specification["source_range"]
    lower, upper = specification["target_range"]
    if not np.isfinite([low, high, lower, upper]).all() or low >= high:
        raise ValueError("invalid source range")
    if [lower, upper] != [preprocess["lower_dn"], preprocess["upper_dn"]]:
        raise ValueError("destination window differs")
    if not 0 <= lower < upper <= 65535:
        raise ValueError("invalid uint16 destination range")
    tolerance = max(1.0, abs(low), abs(high)) * 1e-12
    if a.min() < low - tolerance or a.max() > high + tolerance:
        raise ValueError("image outside declared procedural range; units may differ")
    encoded = lower + np.clip((a - low) / (high - low), 0, 1) * (upper - lower)
    return np.rint(encoded).astype(np.uint16)
