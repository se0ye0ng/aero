"""A recorded run's pinned sources verify against the tree or a declared commit, never by
rewriting the recorded digest."""

import json
import subprocess
from pathlib import Path

import pytest

from aero_ir.utils.manifest import SOURCE_SUPERSESSION, file_sha256, verify_recorded_sources


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    source = tmp_path / "probe.py"
    source.write_text('ROOT = "/local/path"\n')
    _git(tmp_path, "add", "probe.py")
    _git(tmp_path, "commit", "-qm", "original")
    return tmp_path


def _record(repo: Path, commit: str, paths: list[str]) -> None:
    target = repo / SOURCE_SUPERSESSION
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"entries": [{"commit": commit, "paths": paths}]}))


def test_unchanged_source_verifies_from_the_working_tree(repo: Path):
    sources = {"probe.py": file_sha256(repo / "probe.py")}
    assert verify_recorded_sources(sources, repository_root=repo) == {"probe.py": "worktree"}


def test_edited_source_verifies_from_the_declared_commit(repo: Path):
    recorded = file_sha256(repo / "probe.py")
    commit = _git(repo, "rev-parse", "HEAD")
    (repo / "probe.py").write_text('ROOT = root_from_env()\n')
    _record(repo, commit, ["probe.py"])

    assert verify_recorded_sources({"probe.py": recorded}, repository_root=repo) == {
        "probe.py": commit
    }


def test_edited_source_without_a_record_is_rejected(repo: Path):
    recorded = file_sha256(repo / "probe.py")
    (repo / "probe.py").write_text('ROOT = root_from_env()\n')

    with pytest.raises(ValueError, match="matches neither"):
        verify_recorded_sources({"probe.py": recorded}, repository_root=repo)


def test_a_record_naming_another_path_does_not_excuse_this_one(repo: Path):
    recorded = file_sha256(repo / "probe.py")
    commit = _git(repo, "rev-parse", "HEAD")
    (repo / "probe.py").write_text('ROOT = root_from_env()\n')
    _record(repo, commit, ["unrelated.py"])

    with pytest.raises(ValueError, match="matches neither"):
        verify_recorded_sources({"probe.py": recorded}, repository_root=repo)


def test_a_digest_matching_no_revision_is_rejected(repo: Path):
    commit = _git(repo, "rev-parse", "HEAD")
    _record(repo, commit, ["probe.py"])

    with pytest.raises(ValueError, match="matches neither"):
        verify_recorded_sources({"probe.py": "0" * 64}, repository_root=repo)


def test_a_missing_source_is_rejected(repo: Path):
    recorded = file_sha256(repo / "probe.py")
    (repo / "probe.py").unlink()

    with pytest.raises(ValueError, match="matches neither"):
        verify_recorded_sources({"probe.py": recorded}, repository_root=repo)


# Runs that pin a source revision which was never committed: the file was edited between the
# run and the first commit that contains it, so the exact revision that produced the run is not
# recoverable from git and the recorded digest can no longer be checked against anything. This
# predates the public release and is listed rather than hidden. The list must not grow -- a new
# entry means a run's provenance was lost -- and an entry that becomes verifiable must be
# removed. The affected source is named beside each run.
UNRECOVERABLE_SOURCE_RUNS = {
    "experiments/detector_free_train2_01/report.json",  # probe_detector_free_matching.py
    "experiments/registration_mi_probe_train16_01/report.json",  # probe_registration_mi.py
    "experiments/registration_mi_probe_train16_02/report.json",  # probe_registration_mi.py
    "experiments/registration_mind_probe_train160_fine_01/report.json",  # probe_…_mind.py
    "experiments/registration_mind_probe_train16_01/report.json",  # probe_registration_mind.py
    "experiments/registration_ngcc_train16_01/report.json",  # probe_registration_ngcc.py
    "experiments/registration_part_practice_v1/manifest.json",  # registration_practice_ko.md
    "experiments/registration_sam_pareto_train16_01/report.json",  # probe_…_sam_pareto.py
    "experiments/registration_sam_v2_cached_repair_01/report.json",  # probe_registration_sam_v2.py
    "experiments/xoftr_overlay_train16_01/report.json",  # probe_xoftr_overlay.py
}


def _reports_pinning_sources():
    for path in sorted(Path("experiments").rglob("*.json")):
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        sources = report.get("sources")
        if not isinstance(sources, dict) or not sources:
            continue
        if all(isinstance(v, str) and len(v) == 64 for v in sources.values()):
            yield path, sources


def test_every_recorded_report_in_this_repository_verifies():
    """The supersession record covers every run whose sources were edited."""
    reports = list(_reports_pinning_sources())
    if not reports:
        pytest.skip("local experiment artifacts unavailable")

    unverifiable = []
    for path, sources in reports:
        if path.as_posix() in UNRECOVERABLE_SOURCE_RUNS:
            continue
        try:
            verify_recorded_sources(sources)
        except ValueError as error:
            unverifiable.append(f"{path}: {error}")

    assert not unverifiable, "\n".join(unverifiable)


def test_the_unrecoverable_list_stays_accurate():
    """An entry that starts verifying again must be removed from the exception list."""
    reports = dict(_reports_pinning_sources())
    if not reports:
        pytest.skip("local experiment artifacts unavailable")

    present = {p.as_posix() for p in reports}
    for name in UNRECOVERABLE_SOURCE_RUNS & present:
        with pytest.raises(ValueError):
            verify_recorded_sources(reports[Path(name)])
