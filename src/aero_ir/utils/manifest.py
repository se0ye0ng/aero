"""Run manifest - the unit of reproducibility.

``make verify RUN=<id>`` re-executes from this manifest and diffs the metrics. A run that
does not reproduce within tolerance is marked and excluded from reported aggregates.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class RunManifest:
    run_id: str
    config_hash: str
    experiment: str
    seed: int
    dataset_checksums: dict[str, str] = field(default_factory=dict)
    git_sha: str = field(default_factory=_git_sha)
    python: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metrics: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)
