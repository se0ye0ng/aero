import copy

import numpy as np
import pytest

from aero_ir.data.preprocess import (
    apply_uint16_linear_preprocess,
    fit_uint16_linear_preprocess,
    verify_preprocess_spec,
)


class _FixtureDataset:
    def __init__(self):
        self.images = [
            np.arange(16, dtype=np.uint16).reshape(4, 4),
            np.arange(16, 32, dtype=np.uint16).reshape(4, 4),
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], {"image_id": index + 10}


def test_fitted_preprocess_is_deterministic_and_three_channel():
    dataset = _FixtureDataset()
    first = fit_uint16_linear_preprocess(
        dataset,
        manifest_sha256="abc",
        sample_size=2,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )
    second = fit_uint16_linear_preprocess(
        dataset,
        manifest_sha256="abc",
        sample_size=2,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert first == second
    output = apply_uint16_linear_preprocess(dataset.images[0], first)
    assert output.dtype == np.float32
    assert output.shape == (4, 4, 3)
    assert output.min() == 0
    assert output.max() < 255
    assert np.array_equal(output[:, :, 0], output[:, :, 2])


def test_preprocess_rejects_tampering_or_wrong_manifest():
    specification = fit_uint16_linear_preprocess(
        _FixtureDataset(),
        manifest_sha256="abc",
        sample_size=2,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )
    with pytest.raises(ValueError, match="different manifest"):
        verify_preprocess_spec(specification, expected_manifest_sha256="other")

    changed = copy.deepcopy(specification)
    changed["upper_dn"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_preprocess_spec(changed)
