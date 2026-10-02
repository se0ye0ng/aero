import pytest

from scripts.run_grid import deduplicate_zero_generated, expand, main


def test_execution_is_not_silently_reported_as_success(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_grid.py", "e1_reproduce", "--configs", "/not-read"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "No experiment was launched" in capsys.readouterr().err


def test_dry_run_still_previews_without_launching(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "experiment"
    folder.mkdir()
    (folder / "test.yaml").write_text("sweep:\n  seed: [0, 1]\n")
    monkeypatch.setattr(
        "sys.argv", ["run_grid.py", "test", "--configs", str(tmp_path), "--dry-run"]
    )
    main()
    output = capsys.readouterr().out
    assert "test: 2 runs" in output
    assert "seed=0" in output and "seed=1" in output


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
