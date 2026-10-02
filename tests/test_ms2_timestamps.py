import numpy as np
import pytest

from aero_ir.data.ms2_timestamps import compare_timestamps, parse_timestamps


def test_integer_precision_preserved():
    first = 1628215174691250698
    source = parse_timestamps(f"{first}\n{first + 33_333_333}\n".encode(), expected_count=2)
    result = compare_timestamps(source, source + 1)
    assert int(source[0]) == first
    assert result["absolute_skew"]["max_ms"] == 1e-6
    assert result["target_minus_source"]["min_ms"] == 1e-6
    assert result["pairs_modified"] is False
    assert result["exposure_synchronization_verified"] is False
    assert result["registration_qualified"] is False


@pytest.mark.parametrize(
    "data", [b"1.0\n", b"1e9\n", b"-1\n", b"0\n", b"9223372036854775808\n", b"1 2\n", b"\n"]
)
def test_invalid_timestamp(data):
    with pytest.raises(ValueError):
        parse_timestamps(data, expected_count=1)


def test_count_mismatch():
    with pytest.raises(ValueError, match="count"):
        parse_timestamps(b"1\n2\n", expected_count=3)


def test_nonincreasing_and_negative_skew_retained():
    source = np.array([100, 100, 99], dtype=np.int64)
    result = compare_timestamps(source, source - 1)
    assert result["source_nonincreasing_steps"] == 2
    assert result["target_nonincreasing_steps"] == 2
    assert result["target_minus_source"]["min_ms"] == -1e-6
    assert result["absolute_skew"]["max_ms"] == 1e-6
    assert source.tolist() == [100, 100, 99]


def test_single_frame():
    result = compare_timestamps(np.array([1], dtype=np.int64), np.array([1], dtype=np.int64))
    assert result["source_steps"] == {"count": 0}


def test_reject_float_and_mismatched_arrays():
    with pytest.raises(ValueError):
        compare_timestamps(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        compare_timestamps(np.array([1], dtype=np.int64), np.array([1, 2], dtype=np.int64))


def test_extreme_positive_int64_differences_do_not_overflow():
    source = np.array([1, np.iinfo(np.int64).max], dtype=np.int64)
    result = compare_timestamps(source, source[::-1])
    assert result["absolute_skew"]["min_ms"] > 0
    assert result["target_minus_source"]["min_ms"] < 0
