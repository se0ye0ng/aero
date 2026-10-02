import numpy as np
import pytest

from scripts.analyze_roma_confidence import summarize_control


def test_fixed_bins_and_unsupported_count_as_failures():
    result = summarize_control([0, 1, np.nan, 4, 3], [0, 0.1, 0.3, 0.5, 1])
    assert result["queries"] == 5
    assert result["supported"] == 4
    assert result["pck3_all"] == 0.6
    assert [b["queries"] for b in result["bins"]] == [1, 1, 1, 2]
    assert [b["correct"] for b in result["bins"]] == [1, 1, 0, 1]


def test_empty_bin_is_not_perfect_accuracy():
    result = summarize_control([np.nan], [0.01])
    assert result["pck3_all"] == 0
    assert result["median_supported_error_px"] is None
    assert result["bins"][1]["pck3_all"] is None


@pytest.mark.parametrize("errors,confidence", [([], []), ([1], [2]), ([1], [np.nan])])
def test_invalid_inputs(errors, confidence):
    with pytest.raises(ValueError):
        summarize_control(errors, confidence)
