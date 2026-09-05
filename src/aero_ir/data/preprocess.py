"""Train-only, content-addressed intensity preprocessing for thermal detectors."""

from __future__ import annotations

import hashlib
import json

import numpy as np

PREPROCESS_SCHEMA = 1


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> int:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("cannot estimate a quantile from an empty histogram")
    rank = quantile * (total - 1)
    return int(np.searchsorted(np.cumsum(histogram), rank, side="right"))


def fit_uint16_linear_preprocess(
    dataset,
    *,
    manifest_sha256: str,
    sample_size: int = 512,
    seed: int = 0,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> dict:
    """Fit one global linear window from a deterministic training-only sample."""
    if sample_size <= 0 or sample_size > len(dataset):
        raise ValueError(f"sample_size must be in [1, {len(dataset)}]")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    rng = np.random.default_rng(seed)
    indices = sorted(int(index) for index in rng.choice(len(dataset), sample_size, replace=False))
    histogram = np.zeros(65536, dtype=np.int64)
    image_ids: list[int] = []
    for index in indices:
        image, target = dataset[index]
        if image.dtype != np.uint16 or image.ndim != 2:
            raise ValueError(f"analytics16 image {index} must be 2-D uint16, got {image.dtype}")
        histogram += np.bincount(image.ravel(), minlength=65536)
        image_ids.append(int(target["image_id"]))

    lower_dn = _histogram_quantile(histogram, lower_quantile)
    upper_dn = _histogram_quantile(histogram, upper_quantile)
    if lower_dn >= upper_dn:
        raise ValueError(f"degenerate fitted intensity window: [{lower_dn}, {upper_dn}]")
    specification = {
        "schema_version": PREPROCESS_SCHEMA,
        "method": "global_train_histogram_percentile_linear",
        "source_manifest_sha256": manifest_sha256,
        "fit_split": "train",
        "sample_size": sample_size,
        "seed": seed,
        "sample_image_ids": image_ids,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "lower_dn": lower_dn,
        "upper_dn": upper_dn,
        "output_min": 0.0,
        "output_max": 255.0,
        "channel_policy": "repeat_grayscale_to_three_channels",
    }
    specification["preprocess_sha256"] = _canonical_hash(specification)
    verify_preprocess_spec(specification, expected_manifest_sha256=manifest_sha256)
    return specification


def verify_preprocess_spec(
    specification: dict,
    expected_manifest_sha256: str | None = None,
) -> None:
    if specification.get("schema_version") != PREPROCESS_SCHEMA:
        raise ValueError("unsupported preprocessing schema")
    recorded = specification.get("preprocess_sha256")
    unsigned = {key: value for key, value in specification.items() if key != "preprocess_sha256"}
    if not isinstance(recorded, str) or _canonical_hash(unsigned) != recorded:
        raise ValueError("preprocessing specification hash mismatch")
    if (
        expected_manifest_sha256 is not None
        and specification.get("source_manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("preprocessing specification was fit against a different manifest")
    if int(specification["lower_dn"]) >= int(specification["upper_dn"]):
        raise ValueError("preprocessing intensity window is degenerate")


def apply_uint16_linear_preprocess(image: np.ndarray, specification: dict) -> np.ndarray:
    """Map analytics16 to float32 [0, 255] and repeat grayscale into three channels."""
    verify_preprocess_spec(specification)
    array = np.asarray(image)
    if array.dtype != np.uint16 or array.ndim != 2:
        raise ValueError("analytics16 preprocessing requires a 2-D uint16 image")
    lower = float(specification["lower_dn"])
    upper = float(specification["upper_dn"])
    scaled = np.clip((array.astype(np.float32) - lower) / (upper - lower), 0.0, 1.0)
    scaled *= float(specification["output_max"])
    return np.repeat(scaled[:, :, None], 3, axis=2)
