"""CPU-only saved-result verification using archived first-party Python source bytes.

Original filenames remain provenance identities; code and source hashes are read
from a separately verified snapshot. Data, model weights and artifacts are still
read and hashed in their original locations. No report or old manifest is edited.
"""

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

from scripts.snapshot_release_source import verify

TARGETS = {
    "antiuav_roma": "scripts.verify_minima_roma",
    "external_roma": "scripts.probe_external_roma",
    "external_confirmation": "scripts.probe_external_roma_confirmation",
    "roma_resolution": "scripts.probe_roma_resolution",
}


class SourceArchive(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, project, snapshot):
        self.project = project.resolve()
        self.source = (snapshot / "source").resolve()
        manifest = json.loads((snapshot / "manifest.json").read_text())
        self.names = {r["path"] for r in manifest["files"]}
        self.loaded = set()
        self.hashed = set()

    def source_path(self, path):
        original = Path(path)
        if not original.is_absolute():
            original = self.project / original
        # Lexical mapping avoids accidentally following current-tree symlinks.
        try:
            relative = original.relative_to(self.project).as_posix()
        except ValueError:
            return original
        if relative in self.names:
            self.hashed.add(relative)
            return self.source / relative
        return original

    def file_sha256(self, path, chunk_size=1024 * 1024):
        result = hashlib.sha256()
        with self.source_path(path).open("rb") as handle:
            while chunk := handle.read(chunk_size):
                result.update(chunk)
        return result.hexdigest()

    def module_path(self, fullname):
        if fullname == "aero_ir" or fullname.startswith("aero_ir."):
            stem = "src/" + fullname.replace(".", "/")
        elif fullname.startswith("scripts."):
            stem = fullname.replace(".", "/")
        else:
            return None
        for candidate in (stem + "/__init__.py", stem + ".py"):
            if candidate in self.names:
                return candidate
        return None

    def find_spec(self, fullname, path=None, target=None):
        relative = self.module_path(fullname)
        if relative is None:
            if fullname == "aero_ir" or fullname.startswith(("aero_ir.", "scripts.")):
                raise ModuleNotFoundError(f"first-party module missing from snapshot: {fullname}")
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            self.project / relative,
            loader=self,
            submodule_search_locations=[str((self.project / relative).parent)]
            if relative.endswith("/__init__.py")
            else None,
        )

    def create_module(self, spec):
        return None

    def get_code(self, fullname):
        relative = self.module_path(fullname)
        if relative is None:
            raise ImportError(fullname)
        self.loaded.add(relative)
        return compile((self.source / relative).read_bytes(), str(self.project / relative), "exec")

    def exec_module(self, module):
        exec(self.get_code(module.__name__), module.__dict__)
        if module.__name__ == "aero_ir.utils.manifest":
            module.file_sha256 = self.file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--kind", required=True, choices=TARGETS)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    proof = verify(args.snapshot)
    if proof["manifest_sha256"] != args.expected_manifest_sha256:
        raise ValueError("snapshot manifest differs from supplied identity")
    # A clean interpreter is required: do not mix imported current and archived code.
    loaded = [
        n
        for n in sys.modules
        if n == "aero_ir"
        or n.startswith("aero_ir.")
        or (
            n.startswith("scripts.")
            and n not in ("scripts.snapshot_release_source", "scripts.verify_from_source_snapshot")
        )
    ]
    if loaded:
        raise RuntimeError(f"already imported first-party modules: {loaded}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    archive = SourceArchive(Path.cwd(), args.snapshot)
    sys.meta_path.insert(0, archive)
    module = TARGETS[args.kind]
    sys.argv = (
        [module, "--report", str(args.report)]
        if args.kind == "antiuav_roma"
        else [module, "verify", "--out-dir", str(args.report.parent)]
    )
    runpy.run_module(module, run_name="__main__")
    if verify(args.snapshot)["manifest_sha256"] != proof["manifest_sha256"]:
        raise ValueError("snapshot changed during verification")
    print(
        json.dumps(
            dict(
                snapshot_manifest_sha256=proof["manifest_sha256"],
                archived_modules_executed=sorted(archive.loaded),
                archived_source_hash_reads=sorted(archive.hashed),
                verification_source="snapshot",
                neural_inference_replayed=False,
                registration_qualified=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
