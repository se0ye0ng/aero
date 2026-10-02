import pytest

from scripts.probe_roma_resolution import resolution_factory


def test_only_resolution_changes():
    options = dict(
        device="cuda", weights="frozen", dinov2_weights="frozen_dino", amp_dtype="float32"
    )
    construct = resolution_factory(lambda **kwargs: kwargs)
    result = construct(**options)
    assert result == dict(options, coarse_res=840, upsample_res=1152)
    assert options["weights"] == "frozen"


def test_no_silent_override():
    with pytest.raises(ValueError):
        resolution_factory(lambda **kwargs: kwargs)(coarse_res=560)
