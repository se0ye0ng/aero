import hashlib
import json
from types import ModuleType

import pytest

from scripts.verify_from_source_snapshot import SourceArchive


def test_executes_archived_bytes_with_original_identity(tmp_path):
    root, snapshot = tmp_path / "repo", tmp_path / "snapshot"
    (root / "scripts").mkdir(parents=True)
    (snapshot / "source/scripts").mkdir(parents=True)
    name = "scripts/example.py"
    (root / name).write_text("value = 'current'\n")
    original = b"value = 'archived'\n"
    (snapshot / "source" / name).write_bytes(original)
    (snapshot / "manifest.json").write_text(json.dumps(dict(files=[dict(path=name)])))
    loader = SourceArchive(root, snapshot)
    module = ModuleType("scripts.example")
    loader.exec_module(module)
    assert module.value == "archived"
    assert loader.get_code(module.__name__).co_filename == str(root / name)
    assert loader.file_sha256(root / name) == hashlib.sha256(original).hexdigest()
    assert loader.loaded == {name}


def test_non_source_artifact_hash_is_not_substituted(tmp_path):
    root, snapshot = tmp_path / "repo", tmp_path / "snapshot"
    root.mkdir()
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text('{"files": []}')
    data = root / "report.json"
    data.write_bytes(b"actual artifact")
    loader = SourceArchive(root, snapshot)
    assert loader.file_sha256(data) == hashlib.sha256(b"actual artifact").hexdigest()
    assert loader.module_path("numpy") is None
    with pytest.raises(ModuleNotFoundError, match="missing from snapshot"):
        loader.find_spec("aero_ir.missing")
