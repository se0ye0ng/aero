"""Fetch only pinned, public MS2 metadata; never execute downloaded Python.

The images require the authors' access process and are not fetched here. Metadata
stays in ignored experiments/. A Git blob identity is checked before each write.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPOSITORY = "UkcheolShin/MS2-MultiSpectralStereoDataset"
TREE = "0bb0fc1a544d8ad58c7fda5bc025cb0cbb400284"
BLOBS = {
    "MS2dataset/readme.txt": "18c91a64f636284e7f263580c10269c04e627f37",
    "MS2dataset/train_list.txt": "baeeb93027fd0ff3d47b9eaf88896601ffdb6ceb",
    "MS2dataset/val_list.txt": "4b9f05f54e1dbd07f80dc62db703aab72c2d9b21",
    "MS2dataset/test_day_list.txt": "bfa0777c18b492df501342c84f77d1f9058948f3",
    "MS2dataset/test_night_list.txt": "7ab46f8b4cfeed05c473468339c3371f2c23bf46",
    "MS2dataset/test_rainy_list.txt": "2a5854c08764ddd2012853af7d202da151d2e96b",
    "README.md": "96bd975cdc28525ba387b474d692248481ead4da",
    "dataloader/MS2_dataset.py": "ba17cab147eccb50d1d214a936c9403f66589530",
    "utils/utils.py": "d13053a6349a8e5f3bd78a09a0a21ab718b475ba",
}


def verify_blob(data: bytes, expected: str) -> None:
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != expected:
        raise ValueError("metadata differs from pinned Git blob")


def fetch(root: Path) -> dict:
    files = []
    for name, expected in BLOBS.items():
        path = root / name
        url = f"https://api.github.com/repos/{REPOSITORY}/git/blobs/{expected}"
        if path.exists():
            data = path.read_bytes()
        else:
            with urlopen(url, timeout=30) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("metadata response too large")
            payload = json.loads(raw)
            if payload.get("encoding") != "base64":
                raise ValueError("unexpected Git blob encoding")
            data = base64.b64decode("".join(payload["content"].split()), validate=True)
        verify_blob(data, expected)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
        files.append(dict(path=name, git_blob=expected, bytes=len(data), url=url,
                          sha256=hashlib.sha256(data).hexdigest()))
        print(f"verified {name}: {len(data)} bytes", flush=True)
    manifest = dict(schema="ms2_public_metadata_v1", repository=REPOSITORY,
                    git_tree=TREE, files=files, pixels_downloaded=False,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = root / "metadata_manifest.json"
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError("existing manifest differs; use a new output directory")
    else:
        with target.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/external/ms2_metadata"))
    args = parser.parse_args()
    fetch(args.output)


if __name__ == "__main__":
    main()
