"""Content-addressed run manifests and explicit replay verification."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

RUN_MANIFEST_SCHEMA = 1


def canonical_hash(value: object) -> str:
    """Hash JSON-compatible content independently of indentation or key order."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode != 0 or bool(completed.stdout.strip())
    except OSError:
        return None


def _environment() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
    }


def _hardware() -> dict[str, str | int | None]:
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


@dataclass
class RunManifest:
    """Reproducibility record written once a run has a resolved configuration."""

    run_id: str
    config_hash: str
    experiment: str
    seed: int
    dataset_checksums: dict[str, str] = field(default_factory=dict)
    resolved_config: dict = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    command_environment: dict[str, str] = field(default_factory=dict)
    unset_environment: list[str] = field(default_factory=list)
    working_directory: str | None = None
    metrics_path: str | None = None
    replay_metrics_path: str | None = None
    metrics: dict = field(default_factory=dict)
    artifacts: dict[str, dict[str, str]] = field(default_factory=dict)
    environment: dict = field(default_factory=_environment)
    hardware: dict = field(default_factory=_hardware)
    git_sha: str = field(default_factory=_git_sha)
    git_dirty: bool | None = field(default_factory=_git_dirty)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None

    def save(self, path: str | Path) -> str:
        destination = Path(path)
        payload = {"schema_version": RUN_MANIFEST_SCHEMA, **asdict(self)}
        manifest_hash = canonical_hash(payload)
        payload["manifest_sha256"] = manifest_hash
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_hash


def load_manifest(path: str | Path) -> dict:
    """Load a manifest and reject schema or content-hash changes."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported run manifest schema: {manifest.get('schema_version')}")
    recorded_hash = manifest.get("manifest_sha256")
    if not isinstance(recorded_hash, str):
        raise ValueError("run manifest has no manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_hash = canonical_hash(unsigned)
    if actual_hash != recorded_hash:
        raise ValueError(f"run manifest hash mismatch: expected {recorded_hash}, got {actual_hash}")
    resolved_config = manifest.get("resolved_config", {})
    if resolved_config and canonical_hash(resolved_config) != manifest.get("config_hash"):
        raise ValueError("resolved config does not match config_hash")
    return manifest


def _resolved_path(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _git_blob_sha256(path: Path, repository_root: Path, git_sha: str) -> str | None:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        return None
    repository_root = repository_root.resolve()
    try:
        relative = path.resolve().relative_to(repository_root)
    except ValueError:
        return None
    try:
        completed = subprocess.run(
            ["git", "show", f"{git_sha}:{relative.as_posix()}"],
            cwd=repository_root,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


SOURCE_SUPERSESSION = Path("docs/source_supersession.json")


def _supersession_commits(record: Path, name: str) -> list[str]:
    """Commits recorded as still matching ``name`` after it was edited."""
    try:
        entries = json.loads(record.read_text())["entries"]
    except (OSError, KeyError, json.JSONDecodeError):
        return []
    return [e["commit"] for e in entries if name in e.get("paths", ())]


def verify_recorded_sources(
    sources: dict[str, str],
    *,
    repository_root: Path | None = None,
    supersession: Path | None = None,
) -> dict[str, str]:
    """Verify the sources a recorded run pinned.

    A source normally still matches the working tree. When it was edited after the run,
    it is accepted only if it matches at a commit declared in the supersession record,
    and the returned value is that commit instead of ``"worktree"``. A recorded digest is
    never rewritten, so a source that matches nothing raises.
    """
    root = (repository_root or Path.cwd()).resolve()
    record = supersession if supersession is not None else root / SOURCE_SUPERSESSION
    verified: dict[str, str] = {}
    for name, digest in sources.items():
        path = root / name
        if path.is_file() and file_sha256(path) == digest:
            verified[name] = "worktree"
            continue
        for commit in _supersession_commits(record, name):
            if _git_blob_sha256(path, root, commit) == digest:
                verified[name] = commit
                break
        else:
            raise ValueError(
                f"source {name!r} matches neither the working tree nor any commit "
                f"declared in {record}; expected {digest}"
            )
    return verified


def verify_artifacts(
    manifest: dict,
    base: Path,
    *,
    repository_root: Path | None = None,
    git_sha: str = "",
    verified_from_git: list[str] | None = None,
) -> dict[str, str]:
    """Verify every content-addressed input/output declared by a run."""
    verified: dict[str, str] = {}
    for name, specification in manifest.get("artifacts", {}).items():
        if set(specification) != {"path", "sha256"}:
            raise ValueError(f"artifact {name!r} must contain only path and sha256")
        path = _resolved_path(specification["path"], base)
        actual = file_sha256(path) if path.is_file() else None
        if actual == specification["sha256"]:
            verified[name] = actual
            continue
        git_actual = None
        if name.startswith("input__") and repository_root is not None:
            git_actual = _git_blob_sha256(path, repository_root, git_sha)
        if git_actual == specification["sha256"]:
            verified[name] = git_actual
            if verified_from_git is not None:
                verified_from_git.append(name)
            continue
        if actual is None:
            raise FileNotFoundError(path)
        if actual != specification["sha256"]:
            raise ValueError(
                f"artifact {name!r} hash mismatch: expected {specification['sha256']}, got {actual}"
            )
    return verified


def _compare_metrics(expected, actual, tolerance: float, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{prefix or 'metrics'}: expected mapping"]
        for key, value in expected.items():
            child = f"{prefix}.{key}" if prefix else key
            if key not in actual:
                failures.append(f"{child}: missing")
            else:
                failures.extend(_compare_metrics(value, actual[key], tolerance, child))
        return failures
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if expected != actual:
            failures.append(f"{prefix}: expected {expected!r}, got {actual!r}")
        return failures
    if isinstance(expected, int | float):
        try:
            difference = abs(float(expected) - float(actual))
        except (TypeError, ValueError):
            failures.append(f"{prefix}: expected numeric {expected!r}, got {actual!r}")
        else:
            if difference > tolerance:
                failures.append(
                    f"{prefix}: absolute difference {difference:g} exceeds {tolerance:g}"
                )
        return failures
    if expected != actual:
        failures.append(f"{prefix}: expected {expected!r}, got {actual!r}")
    return failures


def verify_run_manifest(
    path: str | Path,
    *,
    tolerance: float = 0.002,
    execute: bool = False,
) -> dict:
    """Verify integrity and optionally replay the manifest's explicit command."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    path = Path(path).resolve()
    manifest = load_manifest(path)
    base = path.parent
    working_directory = manifest.get("working_directory")
    cwd = _resolved_path(working_directory, base) if working_directory else base
    verified_from_git: list[str] = []
    artifacts = verify_artifacts(
        manifest,
        base,
        repository_root=cwd,
        git_sha=str(manifest.get("git_sha", "")),
        verified_from_git=verified_from_git,
    )
    report = {
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "verified_artifacts": artifacts,
        "verified_from_git": verified_from_git,
        "replayed": False,
        "metric_failures": [],
        "ok": True,
    }
    if not execute:
        return report
    if verified_from_git:
        raise ValueError(
            "replay requires the checked-out source files to match the manifest; "
            f"historical Git verification was needed for: {', '.join(verified_from_git)}"
        )

    command = manifest.get("command", [])
    metrics_path = manifest.get("metrics_path")
    if not command or not all(isinstance(token, str) for token in command):
        raise ValueError("replay requires a non-empty command token list")
    if not metrics_path:
        raise ValueError("replay requires metrics_path")
    environment = {**os.environ, **manifest.get("command_environment", {})}
    for name in manifest.get("unset_environment", []):
        if not isinstance(name, str) or not name:
            raise ValueError("unset_environment entries must be non-empty strings")
        environment.pop(name, None)
    replay_metrics_path = manifest.get("replay_metrics_path")
    if replay_metrics_path:
        with tempfile.TemporaryDirectory(prefix=".aero-replay-", dir=base) as replay_root:
            environment["AERO_REPLAY_OUTPUT_ROOT"] = replay_root
            subprocess.run(command, cwd=cwd, env=environment, check=True)
            replayed_path = _resolved_path(replay_metrics_path, Path(replay_root))
            replayed_metrics = json.loads(replayed_path.read_text(encoding="utf-8"))
    else:
        subprocess.run(command, cwd=cwd, env=environment, check=True)
        replayed_path = _resolved_path(metrics_path, cwd)
        replayed_metrics = json.loads(replayed_path.read_text(encoding="utf-8"))
    failures = _compare_metrics(manifest.get("metrics", {}), replayed_metrics, tolerance)
    report.update(
        {
            "replayed": True,
            "metric_failures": failures,
            "ok": not failures,
        }
    )
    return report
