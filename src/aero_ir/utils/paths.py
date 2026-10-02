"""Default dataset locations.

Dataset roots are resolved from the environment so that no contributor's local
filesystem layout is baked into the repository. Every helper returns a relative
path under the data root unless an explicit override is set, and none of them
touches the filesystem: callers validate existence themselves, because the
access scripts in `scripts/download_*.sh` own the layout checks.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "AERO_DATA_ROOT"
DEFAULT_DATA_ROOT = "data"


def data_root() -> Path:
    """Root holding every dataset, overridable with ``AERO_DATA_ROOT``."""
    return Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT))


def dataset_root(name: str, env: str) -> Path:
    """Location of one dataset: ``env`` if set, else ``<data root>/<name>``."""
    override = os.environ.get(env)
    return Path(override) if override else data_root() / name


def antiuav300_root() -> Path:
    return dataset_root("Anti-UAV300", "AERO_ANTIUAV300_ROOT")


def antiuav410_root() -> Path:
    return dataset_root("Anti-UAV410", "AERO_ANTIUAV410_ROOT")


def flir_root() -> Path:
    return dataset_root("FLIR_ADAS_v2", "AERO_FLIR_ROOT")


def ms2_root() -> Path:
    return dataset_root("MS2", "AERO_MS2_ROOT")
