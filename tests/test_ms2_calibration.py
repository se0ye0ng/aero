import io
import pickle

import numpy as np
import pytest

from aero_ir.data.ms2_calibration import read_calibration


def encode(value):
    handle = io.BytesIO()
    np.save(handle, value)
    return handle.getvalue()


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", ">f8", "|u1"])
def test_numeric_dictionary(dtype):
    calibration = {
        "K_rgbL": np.arange(9, dtype=dtype).reshape(3, 3),
        "R_nir2thr": np.asfortranarray(np.eye(3, dtype=dtype)),
        "T_nir2rgb": np.arange(3, dtype=dtype).reshape(3, 1),
    }
    result = read_calibration(encode(calibration))
    for key in calibration:
        np.testing.assert_array_equal(result[key], calibration[key])


def test_disallowed_global_not_called(tmp_path):
    target = tmp_path / "must_not_exist"

    class Attack:
        def __reduce__(self):
            return eval, (f"open({str(target)!r}, 'w').write('bad')",)

    with pytest.raises(ValueError, match="global"):
        read_calibration(encode({"K_rgbL": Attack()}))
    assert not target.exists()


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"K_rgbL": np.zeros((100, 100))},
        {"K_rgbL": np.eye(4)},
        {"K_rgbL": np.eye(3, dtype=np.int64)},
        {"R_nir2thr": np.full((3, 3), np.nan)},
        {"arbitrary": np.eye(3)},
        {"K_rgbL": np.array({"nested": 1}, dtype=object)},
    ],
)
def test_bad_calibration_rejected(value):
    with pytest.raises(ValueError):
        read_calibration(encode(value))


def test_bad_header_and_trailing_payload():
    with pytest.raises(ValueError, match="header"):
        read_calibration(encode(np.eye(3)))
    with pytest.raises(ValueError, match="trailing"):
        read_calibration(encode({"K_rgbL": np.eye(3)}) + b"extra")


def test_legacy_protocol_three():
    # The 2023 release can use NumPy's old module name and pickle protocol3.
    handle = io.BytesIO()
    np.lib.format.write_array_header_1_0(handle, dict(descr="|O", shape=(), fortran_order=False))
    obj = np.empty((), dtype=object)
    obj[()] = {"K_rgbL": np.eye(3)}
    handle.write(pickle.dumps(obj, protocol=3))
    assert read_calibration(handle.getvalue())["K_rgbL"].shape == (3, 3)


def test_extension_opcode_rejected_before_native_unpickler():
    handle = io.BytesIO()
    np.lib.format.write_array_header_1_0(handle, dict(descr="|O", shape=(), fortran_order=False))
    # Even a cached pickle extension must not bypass the global whitelist.
    handle.write(b"\x80\x04\x82\x01.")
    with pytest.raises(ValueError, match="opcode"):
        read_calibration(handle.getvalue())
