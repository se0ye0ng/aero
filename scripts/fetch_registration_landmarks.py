"""Fetch a fixed, small external reference panel; never execute upstream code.

No redistribution license is inferred from public download access. Assets remain
in ignored experiments/. Only project-owned code and aggregate reports may enter
release review. This dataset is not Anti-UAV correspondence ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import yaml

FILES = ("V.JPG", "T.JPG", "points_rgb.txt", "points_thermal.txt")


def get(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=40) as response:
                data = response.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise ValueError("unexpectedly large input")
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


def tree(config, path):
    result = []
    for page in range(1, 101):
        query = urlencode(dict(ref=config["commit"], path=path, page=page, per_page=100))
        entries = json.loads(get(f'{config["api"]}/repository/tree?{query}'))
        result.extend(entries)
        if len(entries) < 100:
            return result
    raise ValueError("unexpected tree pagination")


def select_samples(entries, count):
    if type(count) is not int or count < 1:
        raise ValueError("positive sample count required")
    samples = sorted(
        (e for e in entries if e["type"] == "tree" and e["name"].isdigit()),
        key=lambda e: (int(e["name"]), e["name"]),
    )
    if len(samples) < count:
        raise ValueError("insufficient numeric sample directories")
    return samples[:count]


def verify_git_blob(data, expected):
    header = f"blob {len(data)}\0".encode()
    if hashlib.sha1(header + data).hexdigest() != expected:
        raise ValueError("download does not match pinned Git blob")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/experiment/registration_external_landmarks_cpu.yaml"))
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = yaml.safe_load(raw)
    root = Path(config["cache"])
    root.mkdir(parents=True, exist_ok=True)
    requests, pairs = [], []
    for scene in config["scenes"]:
        for entry in select_samples(tree(config, scene), config["samples_per_scene"]):
            pair_id = entry["path"]
            entries = {e["name"]: e for e in tree(config, pair_id)}
            if not all(name in entries for name in FILES):
                raise ValueError(f"missing required input: {pair_id}")
            pairs.append(pair_id)
            for name in FILES:
                requests.append(entries[name])

    def fetch(entry):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe upstream path")
        target = root / relative
        url = (f'{config["api"]}/repository/files/{quote(entry["path"], safe="")}'
               f'/raw?ref={config["commit"]}')
        data = target.read_bytes() if target.exists() else get(url)
        verify_git_blob(data, entry["id"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
        print(f"verified {relative}: {len(data)} bytes", flush=True)
        return dict(path=entry["path"], git_blob=entry["id"],
                    sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), url=url)

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, requests))
    manifest = dict(kind="external_landmark_inputs_v1", dataset=config["dataset"],
                    commit=config["commit"], pairs=pairs, files=files,
                    config_sha256=hashlib.sha256(raw).hexdigest(),
                    license="no_explicit_dataset_license_identified_do_not_redistribute",
                    qualification="diagnostic_only_no_antiuav_or_generator_approval")
    path = root / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("existing manifest differs; use a fresh cache")
    else:
        with path.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    print(f"wrote/verified {path}")


if __name__ == "__main__":
    main()
