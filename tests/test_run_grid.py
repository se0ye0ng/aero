from scripts.run_grid import deduplicate_zero_generated, expand


def test_zero_generated_controls_are_collapsed_by_declared_factors():
    runs = expand(
        {
            "generator": ["baseline", "learned"],
            "mixing.budget_mode": ["fixed_total", "additive"],
            "mixing.gen_ratio": [0.0, 0.2],
            "seed": [0, 1, 2],
        }
    )

    unique = deduplicate_zero_generated(runs, ["generator", "mixing.budget_mode"])

    zero_controls = [run for run in unique if run["mixing.gen_ratio"] == 0.0]
    generated_runs = [run for run in unique if run["mixing.gen_ratio"] > 0.0]
    assert len(zero_controls) == 3
    assert len(generated_runs) == 12
    assert len(unique) == 15


def test_nonzero_grid_is_unchanged():
    runs = expand({"sensor": ["none", "matched"], "mixing.gen_ratio": [0.2]})

    assert deduplicate_zero_generated(runs, ["sensor"]) == runs
