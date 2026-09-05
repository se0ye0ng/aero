"""The mixing budget is the one place where a silent bug would invalidate every result."""

import pytest

from aero_ir.data.mixing import resolve_mix


def test_fixed_total_holds_total_constant():
    spec = resolve_mix(1000, 1000, gen_ratio=0.2, budget_mode="fixed_total")
    assert spec.n_total == 1000
    assert spec.n_gen == 200
    assert spec.n_real == 800


def test_additive_holds_real_constant():
    spec = resolve_mix(1000, 1000, gen_ratio=0.2, budget_mode="additive")
    assert spec.n_real == 1000
    assert spec.realised_ratio == pytest.approx(0.2, abs=1e-3)


def test_zero_ratio_is_real_only_in_both_modes():
    for mode in ("fixed_total", "additive"):
        spec = resolve_mix(500, 500, gen_ratio=0.0, budget_mode=mode)
        assert spec.n_gen == 0
        assert spec.n_real == 500


def test_insufficient_generated_pool_fails_loudly():
    with pytest.raises(ValueError, match="generated"):
        resolve_mix(1000, 10, gen_ratio=0.5, budget_mode="fixed_total")


def test_ratio_out_of_range():
    with pytest.raises(ValueError):
        resolve_mix(100, 100, gen_ratio=1.5)


def test_unknown_budget_mode():
    with pytest.raises(ValueError, match="budget_mode"):
        resolve_mix(100, 100, gen_ratio=0.1, budget_mode="whatever")


def test_one_ratio_is_rejected_in_additive_mode():
    with pytest.raises(ValueError, match="undefined in additive"):
        resolve_mix(100, 100, gen_ratio=1.0, budget_mode="additive")
