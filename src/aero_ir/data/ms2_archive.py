"""Bounded MS2 archive reading, without executing scripts or calibration pickles."""
from __future__ import annotations

import bz2
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

from aero_ir.utils.manifest import file_sha256


def member_path(member: tarfile.TarInfo, sequence: str) -> PurePosixPath:
    name = member.name
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or not path.parts or path.parts[0] != sequence):
        raise ValueError(f"unsafe or wrong-sequence archive path: {name!r}")
    if not (member.isdir() or member.isfile()):
        raise ValueError(f"unsupported archive member type: {name!r}")
    if member.size < 0 or member.size > 128 * 1024 * 1024:
        raise ValueError(f"unexpected member size: {name!r}")
    return path


def checked_members(archive, sequence):
    seen, total = set(), 0
    for member in archive:
        path = member_path(member, sequence)
        if str(path) in seen:
            raise ValueError(f"duplicate archive member: {path}")
        seen.add(str(path))
        total += member.size
        if len(seen) > 300_000 or total > 150 * 1024**3:
            raise ValueError("archive exceeds inspection limits")
        yield member, path


def pose_errors(data: bytes) -> dict:
    values = np.array([float(v) for v in data.decode("ascii").split()])
    if values.size not in (12, 16) or not np.isfinite(values).all():
        raise ValueError("pose must contain 12 or16 finite numbers")
    pose = values.reshape(-1, 4)
    if pose.shape == (4, 4) and not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("invalid homogeneous pose row")
    rotation = pose[:3, :3]
    return dict(shape=list(pose.shape),
                orthogonality=float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
                determinant=float(abs(np.linalg.det(rotation) - 1)))


def odometry_plan(archive_path: Path, sequence: str, count: int = 16) -> dict:
    if type(count) is not int or count < 2:
        raise ValueError("at least two frame samples required")
    before = file_sha256(archive_path)
    ids, stats = defaultdict(set), {}
    with bz2.open(archive_path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member, path in checked_members(archive, sequence):
                if member.isdir():
                    continue
                if (len(path.parts) != 3 or path.parts[1] not in ("rgb", "nir", "thr", "lidar")
                        or not re.fullmatch(r"\d{6}\.txt", path.name) or member.size > 4096):
                    raise ValueError(f"unexpected odometry member: {path}")
                sensor = path.parts[1]
                ids[sensor].add(path.stem)
                errors = pose_errors(archive.extractfile(member).read())
                item = stats.setdefault(sensor, dict(count=0, shapes={},
                                                    max_orthogonality=0., max_determinant=0.,
                                                    numerical_guard_failures=0,
                                                    first_guard_failure_ids=[]))
                item["count"] += 1
                shape = "x".join(map(str, errors["shape"]))
                item["shapes"][shape] = item["shapes"].get(shape, 0) + 1
                for key in ("orthogonality", "determinant"):
                    item[f"max_{key}"] = max(item[f"max_{key}"], errors[key])
                if errors["orthogonality"] > 1e-5 or errors["determinant"] > 1e-5:
                    # A sampling plan is not a pose-accuracy gate. Retain every
                    # frame ID and report violations; never orthogonalize here.
                    item["numerical_guard_failures"] += 1
                    if len(item["first_guard_failure_ids"]) < 32:
                        item["first_guard_failure_ids"].append(path.stem)
        # Read beyond tar's end marker to verify the bzip2 stream CRC/EOF too.
        while stream.read(1024 * 1024):
            pass
    if set(ids) != {"rgb", "nir", "thr", "lidar"}:
        raise ValueError("missing odometry modality")
    if not all(v == ids["rgb"] for v in ids.values()):
        raise ValueError("odometry frame ID sets differ; do not silently intersect")
    ordered = sorted(ids["rgb"])
    if len(ordered) < count:
        raise ValueError("insufficient frames")
    chosen = [ordered[i * (len(ordered) - 1) // (count - 1)] for i in range(count)]
    if before != file_sha256(archive_path):
        raise ValueError("archive changed during inspection")
    return dict(schema="ms2_train_screen_plan_v1", sequence=sequence, frame_ids=chosen,
                selection="uniform_integer_ranks_in_all_equal_odometry_id_sets",
                odometry_archive_sha256=before, odometry_archive_bytes=archive_path.stat().st_size,
                odometry=stats, all_frame_ids_equal=True, frame_count=len(ordered),
                pose_numerical_guard_tolerance=1e-5,
                pose_numerical_guard_passed=all(
                    s["numerical_guard_failures"] == 0 for s in stats.values()),
                pose_values_modified=False,
                ordered_frame_ids_sha256=hashlib.sha256(json.dumps(ordered).encode()).hexdigest(),
                timestamps_verified=False, physical_registration_qualified=False,
                generator_training_approved=False)


def selected(path: PurePosixPath, kind: str, frames: set[str]) -> bool:
    relative = path.parts[1:]
    if kind == "sync_data":
        if len(relative) == 1:
            return path.suffix.lower() == ".txt" or path.name == "calib.npy"
        if len(relative) == 2 and path.suffix.lower() == ".txt":
            return any(token in path.name.lower() for token in ("timestamp", "format", "readme"))
        if len(relative) != 3 or path.stem not in frames:
            return False
        sensor, folder, _ = relative
        return ((sensor in ("rgb", "thr") and folder in ("img_left", "img_right")
                 and path.suffix == ".png")
                or (sensor == "lidar" and folder in ("left", "right") and path.suffix == ".mat")
                or (sensor == "gps_imu" and folder == "data" and path.suffix == ".txt"))
    if kind == "proj_depth":
        return (len(relative) == 3 and relative[0] in ("rgb", "thr")
                and relative[1] == "depth" and path.suffix == ".png" and path.stem in frames)
    raise ValueError("unsupported archive kind")


def extract_screen(archive_path: Path, output: Path, plan: dict, kind: str,
                   expected_bytes: int) -> dict:
    if kind not in ("sync_data", "proj_depth"):
        raise ValueError("unsupported archive kind")
    if archive_path.stat().st_size != expected_bytes:
        raise ValueError("archive byte count differs; download may be incomplete")
    sequence, frames = plan["sequence"], set(plan["frame_ids"])
    if (not re.fullmatch(r"_\d{4}(?:-\d{2}){5}", sequence) or not frames
            or len(frames) != len(plan["frame_ids"])
            or not all(re.fullmatch(r"\d{6}", f) for f in frames)):
        raise ValueError("invalid screen plan")
    before = file_sha256(archive_path)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    with bz2.open(archive_path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member, path in checked_members(archive, sequence):
                if not member.isfile() or not selected(path, kind, frames):
                    continue
                target = output / kind / str(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.resolve().is_relative_to(output.resolve()):
                    raise ValueError("output path escaped through a symlink")
                digest, copied = hashlib.sha256(), 0
                with archive.extractfile(member) as source, target.open("xb") as sink:
                    while chunk := source.read(1024 * 1024):
                        sink.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                if copied != member.size:
                    raise ValueError(f"truncated archive member: {path}")
                files.append(dict(path=str(target.relative_to(output)), bytes=copied,
                                  sha256=digest.hexdigest()))
        while stream.read(1024 * 1024):
            pass
    if before != file_sha256(archive_path):
        raise ValueError("archive changed during extraction")
    folders = ("img_left", "img_right") if kind == "sync_data" else ("depth",)
    required = {f"{kind}/{sequence}/{s}/{folder}/{f}.png"
                for s in ("rgb", "thr") for folder in folders for f in frames}
    if kind == "sync_data":
        required.add(f"{kind}/{sequence}/calib.npy")
        required.update(f"{kind}/{sequence}/{s}/img_left_timestamp.txt" for s in ("rgb", "thr"))
    missing = sorted(required - {f["path"] for f in files})
    return dict(schema="ms2_selected_extraction_v1", kind=kind, sequence=sequence,
                archive_sha256=before, archive_bytes=expected_bytes, files=files,
                required_missing=missing, complete=not missing,
                calibration_payload_loaded=False, registration_qualified=False)
