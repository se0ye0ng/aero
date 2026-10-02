"""Dataset roots come from the environment, never from a hard-coded local path."""

from pathlib import Path

import pytest

from aero_ir.utils import paths

ROOT_HELPERS = [
    (paths.antiuav300_root, "AERO_ANTIUAV300_ROOT", "Anti-UAV300"),
    (paths.antiuav410_root, "AERO_ANTIUAV410_ROOT", "Anti-UAV410"),
    (paths.flir_root, "AERO_FLIR_ROOT", "FLIR_ADAS_v2"),
    (paths.ms2_root, "AERO_MS2_ROOT", "MS2"),
]
MANAGED_ENV = ["AERO_DATA_ROOT"] + [env for _, env, _ in ROOT_HELPERS]


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for name in MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)


def test_data_root_defaults_to_relative_data_directory():
    assert paths.data_root() == Path("data")


def test_data_root_follows_override(monkeypatch):
    monkeypatch.setenv("AERO_DATA_ROOT", "/mnt/data")
    assert paths.data_root() == Path("/mnt/data")


@pytest.mark.parametrize("helper,env,name", ROOT_HELPERS)
def test_default_is_relative_to_the_data_root(helper, env, name):
    assert helper() == Path("data") / name


@pytest.mark.parametrize("helper,env,name", ROOT_HELPERS)
def test_data_root_override_moves_every_dataset(helper, env, name, monkeypatch):
    monkeypatch.setenv("AERO_DATA_ROOT", "/mnt/data")
    assert helper() == Path("/mnt/data") / name


@pytest.mark.parametrize("helper,env,name", ROOT_HELPERS)
def test_dataset_override_wins_over_the_data_root(helper, env, name, monkeypatch):
    monkeypatch.setenv("AERO_DATA_ROOT", "/mnt/data")
    monkeypatch.setenv(env, "/elsewhere/custom")
    assert helper() == Path("/elsewhere/custom")


@pytest.mark.parametrize("helper,env,name", ROOT_HELPERS)
def test_defaults_are_relative_so_no_local_layout_leaks(helper, env, name):
    assert not helper().is_absolute()


def test_helpers_do_not_require_the_directory_to_exist(monkeypatch):
    monkeypatch.setenv("AERO_ANTIUAV300_ROOT", "/definitely/not/present")
    assert paths.antiuav300_root() == Path("/definitely/not/present")
