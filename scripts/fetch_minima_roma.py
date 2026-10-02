"""Download official pinned RoMa sources and weights without executing upstream code."""

import hashlib
import json
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from aero_ir.utils.manifest import file_sha256

ROOT = Path("experiments/external/minima_roma_0d3fd22")
COMMIT = "0d3fd22dfaf5b7e463a93c3d7f0cbc26df953663"
URLS = {
    "source.tar.gz": f"https://codeload.github.com/LSXI7/RoMa_minima/tar.gz/{COMMIT}",
    "minima_roma.pth": "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth",
    "dinov2_vitl14_pretrain.pth": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
    ),
}


def download(url, target):
    if target.exists():
        raise FileExistsError(f"unmanifested file exists: {target}")
    req = urllib.request.Request(url, headers={"User-Agent": "AERO-research-source-audit"})
    digest, size = hashlib.sha256(), 0
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-") as tmp:
        with urllib.request.urlopen(req, timeout=60) as response:
            expected = response.headers.get("Content-Length")
            while chunk := response.read(8 * 1024 * 1024):
                tmp.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size > 2 * 1024**3:
                    raise ValueError("asset exceeds two GiB bound")
                print(f"{target.name}: {size / 1024**2:.1f} MiB", flush=True)
        if expected is not None and size != int(expected):
            raise ValueError("incomplete download")
        tmp.flush()
        os.fsync(tmp.fileno())
        os.link(tmp.name, target)  # Atomic creation; never overwrite an existing file.
    return dict(url=url, bytes=size, sha256=digest.hexdigest())


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "download_manifest.json"
    recorded = json.loads(manifest.read_text()) if manifest.exists() else {}
    for name, url in URLS.items():
        path = ROOT / name
        if name in recorded:
            entry = recorded[name]
            if entry["url"] != url or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"previous download changed: {path}")
            print(f"verified existing {name}", flush=True)
            continue
        recorded[name] = download(url, path)
        # This manifest is owned by this downloader, never a frozen experiment report.
        with tempfile.NamedTemporaryFile(mode="w", dir=ROOT, delete=False) as tmp:
            json.dump(recorded, tmp, indent=2)
            temporary = Path(tmp.name)
        temporary.replace(manifest)
    source = ROOT / f"RoMa_minima-{COMMIT}"
    if source.exists():
        raise FileExistsError(
            f"source tree already exists; inspect rather than overwrite: {source}"
        )
    with tarfile.open(ROOT / "source.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if (
                not parts
                or parts[0] != source.name
                or ".." in parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
        if sum(m.size for m in members) > 512 * 1024**2:
            raise ValueError("expanded source archive too large")
        archive.extractall(ROOT, members=members, filter="data")
    print(json.dumps(recorded, indent=2))
    print("Source and weights downloaded only; no model code executed.")


if __name__ == "__main__":
    main()
