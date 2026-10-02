"""Preserve current first-party source bytes before release cleanup, not experiment replay."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

TREES = ("src", "scripts", "tests", "configs", "requirements", ".github/workflows")
FILES = ("pyproject.toml", "Makefile", ".pre-commit-config.yaml")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_names(root):
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    )
    names = sorted(set(n.decode() for n in output.split(b"\0") if n))
    return [n for n in names if n in FILES or any(n.startswith(t + "/") for t in TREES)]


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
        raise ValueError("unsafe snapshot path")
    if name not in FILES and not any(name.startswith(t + "/") for t in TREES):
        raise ValueError("path outside first-party source allowlist")
    return path


def create(root, out):
    root = root.resolve()
    names = source_names(root)
    if not names:
        raise ValueError("empty source inventory")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in names:
        relative = safe_relative(name)
        source = root / relative
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
            raise ValueError(f"nonregular or escaped source: {name}")
        data = source.read_bytes()
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(data)
        destination.chmod(source.stat().st_mode & 0o777)
        rows.append(dict(path=name, sha256=digest(data), bytes=len(data)))
    # Detect edits occurring while the snapshot was copied; do not certify a mixed snapshot.
    for row in rows:
        if digest((root / row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError(f"source changed while copying: {row['path']}")
    if source_names(root) != names:
        raise ValueError("source inventory changed while copying")
    content = dict(
        kind="first_party_source_snapshot_v1",
        files=rows,
        scope=list(TREES) + list(FILES),
        omissions=["datasets", "weights", "results", "docs", "git history", "environment"],
        experiment_replayed=False,
        registration_qualified=False,
        limitations="Current source only; not proof of every historical run's source.",
    )
    with (out / "manifest.json").open("x") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
    return verify(out)


def verify(out):
    manifest_path = out / "manifest.json"
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if (
        manifest["kind"] != "first_party_source_snapshot_v1"
        or manifest["registration_qualified"]
        or manifest["experiment_replayed"]
    ):
        raise ValueError("invalid snapshot type or claim")
    root = (out / "source").resolve()
    seen = set()
    for row in manifest["files"]:
        name = row["path"]
        if name in seen:
            raise ValueError("duplicate snapshot entry")
        seen.add(name)
        path = root / safe_relative(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("escaped snapshot entry")
        data = path.read_bytes()
        if len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ValueError(f"snapshot bytes differ: {name}")
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()
    }
    if not seen or actual != seen:
        raise ValueError("snapshot inventory differs")
    return dict(
        ok=True,
        files=len(seen),
        manifest_sha256=digest(payload),
        experiment_replayed=False,
        registration_qualified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = create(Path.cwd(), args.out) if args.action == "create" else verify(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
