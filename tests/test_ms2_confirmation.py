import pytest

from scripts.prepare_ms2_confirmation import interval_midpoints


def test_frozen_interval_midpoints_not_existing_endpoints():
    frames = ["000000", "000696", "001392", "010441"]
    result = interval_midpoints(frames)
    assert result == ["000348", "001044", "005916"]
    assert not set(result).intersection(frames)


@pytest.mark.parametrize(
    "frames", [["000002", "000001"], ["000000", "000001"], ["000000"], ["0", "000030"]]
)
def test_invalid_intervals_fail(frames):
    with pytest.raises(ValueError):
        interval_midpoints(frames)
