import hashlib

import numpy as np
import pytest

from scripts.audit_ms2_metadata import inspect_train_sequence, numpy_header, parse_sequences
from scripts.audit_ms2_metadata import split_overlaps as overlaps
from scripts.fetch_ms2_metadata import verify_blob


def test_split_parser_preserves_last_line_without_newline():
    assert parse_sequences("_2021-08-06-10-59-33\n_2021-08-06-11-37-46") == [
        "_2021-08-06-10-59-33",
        "_2021-08-06-11-37-46",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "../train",
        "/test",
        "_2021-08-06-10-59-33 extra",
        "_2021-08-06-10-59-33\n_2021-08-06-10-59-33",
    ],
)
def test_invalid_split_rejected(text):
    with pytest.raises(ValueError):
        parse_sequences(text)


def test_overlaps_are_not_silently_removed():
    assert overlaps({"train": ["a", "b"], "val": ["b"], "test": ["c"]}) == {"train__val": ["b"]}


def test_git_blob_verification():
    data = b"metadata\n"
    blob = hashlib.sha1(b"blob 9\0" + data).hexdigest()
    verify_blob(data, blob)
    with pytest.raises(ValueError, match="pinned"):
        verify_blob(data + b"\n", blob)


def test_header_does_not_load_object_payload(tmp_path):
    path = tmp_path / "calib.npy"
    # Deliberately no payload: only the header is needed and no pickle is executed.
    with path.open("wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle, dict(descr="|O", fortran_order=False, shape=())
        )
    result = numpy_header(path)
    assert result["contains_objects"] is True
    assert result["payload_loaded"] is False


def test_missing_local_data_cannot_look_complete(tmp_path):
    result = inspect_train_sequence(tmp_path, "_2021-08-06-10-59-33")
    assert len(result["missing"]) == 6
    assert result["pixels_decoded"] is False
    assert result["calibration_values_checked"] is False
