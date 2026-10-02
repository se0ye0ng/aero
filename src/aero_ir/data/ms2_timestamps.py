"""Integer timestamp diagnostics; never silently re-pair or approve a dataset."""
from __future__ import annotations

import re

import numpy as np


def parse_timestamps(data: bytes, *, expected_count: int) -> np.ndarray:
    """Read one positive signed-int64 nanosecond timestamp per line.

    Unix nanoseconds exceed float64's exact-integer range. Subtract timestamps
    as integers first and convert only the resulting durations to milliseconds.
    """
    if type(expected_count) is not int or not 1 <= expected_count <= 300_000:
        raise ValueError("invalid expected timestamp count")
    if len(data) > expected_count * 32:
        raise ValueError("timestamp file exceeds size bound")
    lines = data.decode("ascii").splitlines()
    if len(lines) != expected_count:
        raise ValueError("timestamp count does not match the complete frame inventory")
    values = []
    for line in lines:
        if not re.fullmatch(r"[0-9]{1,19}", line):
            raise ValueError("expected one unsigned decimal integer per line")
        value = int(line)
        if not 0 < value <= np.iinfo(np.int64).max:
            raise ValueError("timestamp outside positive int64 range")
        values.append(value)
    return np.asarray(values, dtype=np.int64)


def _durations(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0}
    milliseconds = values.astype(np.float64) / 1_000_000
    return dict(count=len(values), min_ms=float(milliseconds.min()),
                median_ms=float(np.median(milliseconds)),
                p95_ms=float(np.percentile(milliseconds, 95)),
                max_ms=float(milliseconds.max()))


def compare_timestamps(source: np.ndarray, target: np.ndarray) -> dict:
    """Describe original same-index pairs, not corrected/exposure-certified pairs.

    A common nanosecond clock is a hypothesis supplied by the source format.
    Timestamp agreement alone cannot certify exposure synchronization or geometry.
    """
    source, target = np.asarray(source), np.asarray(target)
    if (source.ndim != 1 or source.shape != target.shape or len(source) == 0
            or source.dtype != np.int64 or target.dtype != np.int64
            or (source <= 0).any() or (target <= 0).any()):
        raise ValueError("equal nonempty positive int64 timestamp arrays required")
    signed = target - source
    source_steps, target_steps = np.diff(source), np.diff(target)
    return dict(
        frame_count=len(source), clock_unit_assumption="nanoseconds_shared_clock",
        paired_by="unchanged_frame_index", pairs_modified=False,
        target_minus_source=_durations(signed), absolute_skew=_durations(np.abs(signed)),
        source_steps=_durations(source_steps), target_steps=_durations(target_steps),
        source_nonincreasing_steps=int(np.count_nonzero(source_steps <= 0)),
        target_nonincreasing_steps=int(np.count_nonzero(target_steps <= 0)),
        exposure_synchronization_verified=False, registration_qualified=False)
