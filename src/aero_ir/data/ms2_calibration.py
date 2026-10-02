"""Read small NumPy calibration dictionaries without invoking pickle globals.

The restricted reader substitutes inert containers for NumPy pickle constructors.
Only after checking small shapes and byte lengths does it create numeric arrays.
This checks serialization, not calibration accuracy or coordinate conventions.
"""
from __future__ import annotations

import io
import pickle
import pickletools
import re

import numpy as np


class _DType:
    def __init__(self, code, align=False, copy=False):
        if (not isinstance(code, str) or not re.fullmatch(r"[<>=|]?(?:f[48]|u1|O[48]?)", code)
                or type(align) is not bool or type(copy) is not bool):
            raise ValueError("unsupported calibration dtype")
        self.code = code
        self.byteorder = "="

    def __setstate__(self, state):
        if (not isinstance(state, tuple) or len(state) != 8 or state[0] != 3
                or state[1] not in ("<", ">", "=", "|")
                or state[2:5] != (None, None, None)
                or state[5:7] != (-1, -1) or type(state[7]) is not int
                or state[7] not in (0, 63)):
            raise ValueError("unsupported calibration dtype state")
        self.byteorder = state[1]


class _Array:
    def __setstate__(self, state):
        if (not isinstance(state, tuple) or len(state) != 5 or state[0] != 1
                or type(state[1]) is not tuple or type(state[2]) is not _DType
                or type(state[3]) is not bool):
            raise ValueError("unsupported calibration array state")
        _, self.shape, self.dtype, self.fortran, self.data = state
        if (len(self.shape) > 2 or any(type(n) is not int or not 1 <= n <= 16
                                       for n in self.shape)):
            raise ValueError("calibration array shape exceeds bound")
        if "O" in self.dtype.code:
            if self.shape != () or type(self.data) is not list or len(self.data) != 1:
                raise ValueError("only scalar dictionary object arrays supported")
        elif type(self.data) is not bytes or len(self.data) > 2048:
            raise ValueError("expected bounded numeric calibration bytes")


def _reconstruct(subtype, shape, dtype):
    if subtype is not _Array or shape != (0,) or dtype != b"b":
        raise ValueError("unsupported calibration reconstruction")
    return _Array()


class _Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in (("numpy.core.multiarray", "_reconstruct"),
                              ("numpy._core.multiarray", "_reconstruct")):
            return _reconstruct
        if module == "numpy" and name == "ndarray":
            return _Array
        if module == "numpy" and name == "dtype":
            return _DType
        raise ValueError(f"unsupported calibration pickle global: {module}.{name}")

    def persistent_load(self, pid):
        raise ValueError("persistent calibration references are not supported")


def read_calibration(data: bytes) -> dict[str, np.ndarray]:
    if not 0 < len(data) <= 1024 * 1024:
        raise ValueError("calibration file exceeds bound")
    handle = io.BytesIO(data)
    version = np.lib.format.read_magic(handle)
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    if version not in readers:
        raise ValueError("unsupported calibration npy version")
    shape, fortran, dtype = readers[version](handle)
    if shape != () or fortran or dtype != np.dtype(object):
        raise ValueError("expected scalar object calibration header")
    payload = handle.read()
    stop = None
    for opcode, _, position in pickletools.genops(payload):
        # Cached extension globals can bypass find_class in the native reader.
        if opcode.name in ("EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID",
                           "NEXT_BUFFER", "READONLY_BUFFER"):
            raise ValueError("unsupported calibration pickle reference opcode")
        if opcode.name == "STOP":
            stop = position
    if stop != len(payload) - 1:
        raise ValueError("trailing calibration payload")
    root = _Reader(io.BytesIO(payload)).load()
    if (type(root) is not _Array or root.shape != () or root.fortran
            or "O" not in root.dtype.code or type(root.data[0]) is not dict):
        raise ValueError("expected scalar calibration dictionary")
    payload = root.data[0]
    if not 1 <= len(payload) <= 64:
        raise ValueError("unexpected calibration key count")
    result = {}
    for key, value in payload.items():
        if (type(key) is not str or not re.fullmatch(r"[KRT]_[a-zA-Z0-9]{1,24}", key)
                or type(value) is not _Array or "O" in value.dtype.code):
            raise ValueError("expected named numeric calibration matrix")
        expected_shapes = ((3, 3),) if key[0] in "KR" else ((3,), (3, 1))
        if value.shape not in expected_shapes:
            raise ValueError(f"invalid matrix shape: {key}")
        numeric_dtype = np.dtype(value.dtype.code).newbyteorder(value.dtype.byteorder)
        elements = int(np.prod(value.shape))
        if len(value.data) != elements * numeric_dtype.itemsize:
            raise ValueError(f"invalid matrix byte length: {key}")
        array = np.frombuffer(value.data, dtype=numeric_dtype).reshape(
            value.shape, order="F" if value.fortran else "C").astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite calibration matrix: {key}")
        result[key] = array
    return result
