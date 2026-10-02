import io
import tarfile

import numpy as np
import pytest

from aero_ir.data.ms2_archive import (
    extract_screen,
    member_path,
    odometry_plan,
    pose_errors,
    selected,
)

SEQ = "_2021-08-06-10-59-33"


def write_tar(path, entries):
    with tarfile.open(path, "w:bz2") as archive:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def identity_bytes(rows=4):
    return " ".join(map(str, np.eye(4)[:rows].ravel())).encode()


@pytest.mark.parametrize(
    "name",
    [
        "/outside",
        "../outside",
        f"{SEQ}/../outside",
        f"{SEQ}/rgb\\outside",
        "different_sequence/file",
    ],
)
def test_archive_paths_fail_closed(name):
    with pytest.raises(ValueError, match="path"):
        member_path(tarfile.TarInfo(name), SEQ)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE])
def test_links_and_devices_are_rejected(kind):
    member = tarfile.TarInfo(f"{SEQ}/calib.npy")
    member.type = kind
    with pytest.raises(ValueError, match="type"):
        member_path(member, SEQ)


def test_pose_formats_and_bad_matrices():
    for rows in (3, 4):
        assert pose_errors(identity_bytes(rows))["shape"] == [rows, 4]
    with pytest.raises(ValueError, match="finite"):
        pose_errors(b"nan " * 16)
    with pytest.raises(ValueError, match="row"):
        pose_errors(b"0 " * 16)
    reflection = np.eye(4)
    reflection[0, 0] = -1
    assert pose_errors(" ".join(map(str, reflection.ravel())).encode())["determinant"] == 2


def test_odometry_plan_preserves_all_modalities_and_freezes_ranks(tmp_path):
    path = tmp_path / "odom.bz2"
    write_tar(
        path,
        [
            (f"{SEQ}/{s}/{i:06d}.txt", identity_bytes())
            for s in ("rgb", "thr", "nir", "lidar")
            for i in range(5)
        ],
    )
    result = odometry_plan(path, SEQ, count=3)
    assert result["frame_ids"] == ["000000", "000002", "000004"]
    assert result["frame_count"] == 5
    assert result["timestamps_verified"] is False
    assert result["physical_registration_qualified"] is False
    assert result["odometry"]["rgb"]["shapes"] == {"4x4": 5}


def test_odometry_missing_frames_are_not_silently_intersected(tmp_path):
    path = tmp_path / "odom.bz2"
    write_tar(
        path,
        [
            (f"{SEQ}/{s}/{i:06d}.txt", identity_bytes())
            for s in ("rgb", "thr", "nir", "lidar")
            for i in range(3)
            if not (s == "thr" and i == 2)
        ],
    )
    with pytest.raises(ValueError, match="differ"):
        odometry_plan(path, SEQ, count=2)


def test_nonrigid_pose_is_reported_without_dropping_or_correcting_frames(tmp_path):
    path = tmp_path / "odom.bz2"
    pose = np.eye(4)
    pose[0, 0] = 1.00002
    data = " ".join(map(str, pose.ravel())).encode()
    write_tar(
        path,
        [
            (f"{SEQ}/{s}/{i:06d}.txt", data)
            for s in ("rgb", "thr", "nir", "lidar")
            for i in range(3)
        ],
    )
    result = odometry_plan(path, SEQ, count=2)
    assert result["pose_numerical_guard_passed"] is False
    assert result["pose_values_modified"] is False
    assert result["frame_count"] == 3
    assert result["odometry"]["rgb"]["numerical_guard_failures"] == 3


def test_truncated_bzip_does_not_produce_plan(tmp_path):
    path = tmp_path / "odom.bz2"
    write_tar(
        path,
        [
            (f"{SEQ}/{s}/{i:06d}.txt", identity_bytes())
            for s in ("rgb", "thr", "nir", "lidar")
            for i in range(3)
        ],
    )
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises((EOFError, OSError, tarfile.ReadError)):
        odometry_plan(path, SEQ, count=2)


def test_select_only_planned_depth_not_filtered_predictions():
    from pathlib import PurePosixPath

    assert selected(PurePosixPath(f"{SEQ}/thr/depth/000001.png"), "proj_depth", {"000001"})
    assert not selected(
        PurePosixPath(f"{SEQ}/thr/depth_filtered/000001.png"), "proj_depth", {"000001"}
    )
    assert not selected(PurePosixPath(f"{SEQ}/thr/depth/000002.png"), "proj_depth", {"000001"})


def test_selective_extraction_checks_completeness_and_preserves_existing_output(tmp_path):
    path = tmp_path / "data.bz2"
    write_tar(
        path,
        [
            (f"{SEQ}/{s}/depth/{i:06d}.png", b"fixture_not_decoded")
            for s in ("rgb", "thr")
            for i in range(3)
        ],
    )
    output = tmp_path / "selected"
    plan = dict(sequence=SEQ, frame_ids=["000000", "000002"])
    with pytest.raises(ValueError, match="byte count"):
        extract_screen(path, output, plan, "proj_depth", path.stat().st_size + 1)
    assert not output.exists()
    result = extract_screen(path, output, plan, "proj_depth", path.stat().st_size)
    assert result["complete"] is True
    assert len(result["files"]) == 4
    assert not list(output.rglob("000001.png"))
    with pytest.raises(FileExistsError):
        extract_screen(path, output, plan, "proj_depth", path.stat().st_size)
    incomplete = extract_screen(
        path,
        tmp_path / "missing",
        dict(sequence=SEQ, frame_ids=["000009"]),
        "proj_depth",
        path.stat().st_size,
    )
    assert incomplete["complete"] is False
    assert len(incomplete["required_missing"]) == 2


def test_duplicate_archive_members_fail(tmp_path):
    path = tmp_path / "duplicate.bz2"
    write_tar(path, [(f"{SEQ}/rgb/depth/000000.png", b"x")] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        extract_screen(
            path,
            tmp_path / "output",
            dict(sequence=SEQ, frame_ids=["000000"]),
            "proj_depth",
            path.stat().st_size,
        )
