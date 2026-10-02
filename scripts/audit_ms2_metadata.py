"""Audit MS2 source metadata without opening held-out pixels or unpickling calibration.

This is an acquisition/preflight report, never registration qualification. The
optional local check inspects one predetermined official training sequence only.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np

from aero_ir.utils.manifest import file_sha256
from scripts.fetch_ms2_metadata import BLOBS, TREE, verify_blob

SPLITS = ("train", "val", "test_day", "test_night", "test_rainy")


def parse_sequences(text: str) -> list[str]:
    result = []
    for line in text.splitlines():
        name = line.strip()
        if not name:
            continue
        if not re.fullmatch(r"_\d{4}(?:-\d{2}){5}", name):
            raise ValueError(f"invalid sequence ID: {name!r}")
        if name in result:
            raise ValueError(f"duplicate sequence ID: {name}")
        result.append(name)
    if not result:
        raise ValueError("empty split")
    return result


def split_overlaps(splits: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        f"{a}__{b}": sorted(set(splits[a]) & set(splits[b]))
        for a, b in itertools.combinations(sorted(splits), 2)
        if set(splits[a]) & set(splits[b])
    }


def numpy_header(path: Path) -> dict:
    """Read only the bounded .npy header; object payload is never deserialized."""
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported calibration .npy version: {version}")
    return dict(shape=list(shape), fortran_order=fortran, dtype=str(dtype),
                contains_objects=dtype.hasobject, payload_loaded=False)


def inspect_train_sequence(root: Path, sequence: str) -> dict:
    base = root / "sync_data" / sequence
    result = dict(sequence=sequence, paths={}, missing=[], inventories={})
    for name in ("calib.npy", "readme.txt"):
        path = base / name
        if not path.is_file():
            result["missing"].append(str(path))
            continue
        item = dict(path=str(path), sha256=file_sha256(path))
        if name == "calib.npy":
            item["header"] = numpy_header(path)
        result["paths"][name] = item
    for sensor in ("rgb", "thr"):
        for kind, directory in (
            ("img_left", base / sensor / "img_left"),
            ("depth", root / "proj_depth" / sequence / sensor / "depth"),
        ):
            names = sorted(p.name for p in directory.glob("*.png") if p.is_file())
            if not names:
                result["missing"].append(str(directory))
            result["inventories"][f"{sensor}/{kind}"] = dict(
                count=len(names), first=names[0] if names else None,
                last=names[-1] if names else None,
                filenames_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),
            )
    result["pixels_decoded"] = False
    result["timestamps_checked"] = False
    result["calibration_values_checked"] = False
    result["file_identity_is_not_temporal_pairing"] = True
    return result


def audit(metadata: Path, data_root: Path | None = None) -> dict:
    hashes = {}
    for name, blob in BLOBS.items():
        data = (metadata / name).read_bytes()
        verify_blob(data, blob)
        hashes[name] = hashlib.sha256(data).hexdigest()
    splits = {
        name: parse_sequences((metadata / "MS2dataset" / f"{name}_list.txt").read_text())
        for name in SPLITS
    }
    overlaps = split_overlaps(splits)
    # Freeze acquisition choice before examining any imagery or model scores.
    first_train = sorted(splits["train"])[0]
    local = inspect_train_sequence(data_root, first_train) if data_root else None
    return dict(
        schema="ms2_acquisition_preflight_v1", upstream_git_tree=TREE,
        input_sha256=hashes,
        source_sha256={str(Path(__file__).name): file_sha256(__file__),
                       "fetch_ms2_metadata.py": file_sha256(Path(__file__).with_name(
                           "fetch_ms2_metadata.py"))},
        splits=splits, sequence_counts={k: len(v) for k, v in splits.items()},
        sequence_overlaps=overlaps, sequence_id_disjoint=not overlaps,
        independent_routes_or_recordings_verified=False,
        first_training_sequence=first_train, local_training_preflight=local,
        status=("metadata_verified_data_not_inspected" if local is None else
                "local_preflight_missing_files" if local["missing"] else
                "local_inventory_complete_geometry_not_evaluated"),
        registration_qualified=False, generator_training_approved=False,
        antiuav_status_changed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        default=Path("experiments/external/ms2_metadata"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out}")
    result = audit(args.metadata, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: result[k] for k in (
        "status", "sequence_counts", "sequence_id_disjoint", "first_training_sequence",
        "registration_qualified", "generator_training_approved",
    )}, indent=2))
    print(f"wrote {args.out}")
    if result["sequence_overlaps"] or (
        result["local_training_preflight"] and result["local_training_preflight"]["missing"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
