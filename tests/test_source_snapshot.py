import json

import pytest

from scripts import snapshot_release_source as snapshot


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src/example.py").write_text("value = 1\n")
    monkeypatch.setattr(snapshot, "source_names", lambda _: ["src/example.py"])
    return root, tmp_path / "snapshot"


def test_preserves_bytes_and_refuses_overwrite(source):
    root, out = source
    result = snapshot.create(root, out)
    assert result["ok"] and not result["experiment_replayed"]
    assert (out / "source/src/example.py").read_bytes() == (root / "src/example.py").read_bytes()
    with pytest.raises(FileExistsError):
        snapshot.create(root, out)


def test_detects_changed_or_extra_file(source):
    root, out = source
    snapshot.create(root, out)
    (out / "source/src/extra.py").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        snapshot.verify(out)
    (out / "source/src/example.py").write_text("changed")
    with pytest.raises(ValueError, match="bytes differ"):
        snapshot.verify(out)


@pytest.mark.parametrize(
    "name", ["../private", "/tmp/file", "docs/inspiration.md", "src/../secret", "src\\secret"]
)
def test_rejects_escape_and_private_scope(name):
    with pytest.raises(ValueError):
        snapshot.safe_relative(name)


def test_rejects_symlink(source):
    root, out = source
    (root / "src/example.py").unlink()
    (root / "src/example.py").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="nonregular"):
        snapshot.create(root, out)


def test_manual_approval_not_allowed(source):
    root, out = source
    snapshot.create(root, out)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["registration_qualified"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="claim"):
        snapshot.verify(out)
