"""Bounded axis-aligned patch profiles; local peaks are not correspondence GT."""

from __future__ import annotations

import numpy as np

from aero_ir.registration.stereo_patch_evidence import patches

OFFSETS = np.arange(-3.0, 3.01, 0.5)


def axis_profiles(left, right, left_support, right_support, xy, disparity):
    if disparity.shape != (len(xy),):
        raise ValueError("one fixed disparity per point required")
    scores = {axis: np.full((len(xy), len(OFFSETS)), np.nan) for axis in ("x", "y")}
    for start in range(0, len(xy), 1024):
        sl = slice(start, start + 1024)
        lp, lv = patches(left, left_support, xy[sl])
        lc = lp - lp.mean(axis=1, keepdims=True)
        ll = np.sum(lc * lc, axis=1)
        origin = xy[sl].copy()
        origin[:, 0] -= disparity[sl]
        for axis, dimension in (("x", 0), ("y", 1)):
            for j, offset in enumerate(OFFSETS):
                if axis == "y" and offset == 0:
                    scores[axis][sl, j] = scores["x"][sl, j]
                    continue
                shifted = origin.copy()
                shifted[:, dimension] += offset
                rp, rv = patches(right, right_support, shifted)
                rc = rp - rp.mean(axis=1, keepdims=True)
                rr = np.sum(rc * rc, axis=1)
                valid = (
                    lv
                    & rv
                    & np.isfinite(disparity[sl])
                    & (disparity[sl] > 0)
                    & (ll > 1e-12)
                    & (rr > 1e-12)
                )
                values = np.full(len(lp), np.nan)
                values[valid] = np.clip(
                    np.sum(lc[valid] * rc[valid], axis=1) / np.sqrt(ll[valid] * rr[valid]),
                    -1.0,
                    1.0,
                )
                scores[axis][sl, j] = values
    return {f"ncc_profile_{axis}": values for axis, values in scores.items()}


def peak_statistics(profile, eligible):
    """Complete-profile comparisons; ties never silently become a negative shift."""
    if profile.shape != (len(eligible), len(OFFSETS)) or eligible.dtype != bool:
        raise ValueError("Nx13 profile and boolean population required")
    if np.any(eligible & ~np.isfinite(profile).all(axis=1)):
        raise ValueError("eligible points must have all offsets observed")
    values = profile[eligible]
    if not len(values):
        return dict(
            points=0,
            unique_peak_points=0,
            tied_peak_points=0,
            boundary_peak_points=0,
            median_unique_offset_px=None,
            median_peak_gain=None,
            median_separated_peak_gap=None,
            offset_counts={str(v): 0 for v in OFFSETS},
        )
    best = values.max(axis=1)
    ties = np.abs(values - best[:, None]) <= 1e-9
    unique = ties.sum(axis=1) == 1
    indices = values.argmax(axis=1)
    offsets = OFFSETS[indices]
    separated = np.abs(OFFSETS[None, :] - offsets[:, None]) >= 1.0
    rival = np.max(np.where(separated, values, -np.inf), axis=1)
    return dict(
        points=len(values),
        unique_peak_points=int(unique.sum()),
        tied_peak_points=int((~unique).sum()),
        boundary_peak_points=int((ties[:, 0] | ties[:, -1]).sum()),
        median_unique_offset_px=float(np.median(offsets[unique])) if unique.any() else None,
        median_peak_gain=float(np.median(best - values[:, len(OFFSETS) // 2])),
        median_separated_peak_gap=float(np.median((best - rival)[unique]))
        if unique.any()
        else None,
        offset_counts={str(v): int((unique & (offsets == v)).sum()) for v in OFFSETS},
    )


def summarize(profiles, left_std):
    x, y = profiles["ncc_profile_x"], profiles["ncc_profile_y"]
    if x.shape != y.shape or x.shape != (len(left_std), len(OFFSETS)):
        raise ValueError("matching profiles and reference population required")
    common = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    masks = dict(
        all=np.ones(len(left_std), bool),
        left_std_below_2=left_std < 2,
        left_std_2_to_8=(left_std >= 2) & (left_std < 8),
        left_std_at_least_8=left_std >= 8,
        left_patch_unavailable=~np.isfinite(left_std),
    )
    return {
        name: dict(
            reference_points=int(mask.sum()),
            complete_profiles=int((mask & common).sum()),
            x=peak_statistics(x, mask & common),
            y=peak_statistics(y, mask & common),
        )
        for name, mask in masks.items()
    }


def aggregate(rows):
    result = []
    for sensor, condition in sorted({(r["sensor"], r["condition"]) for r in rows}):
        selected = [r for r in rows if (r["sensor"], r["condition"]) == (sensor, condition)]
        for name in selected[0]["strata"]:
            stats = [r["strata"][name] for r in selected]
            record = dict(
                sensor=sensor,
                condition=condition,
                stratum=name,
                planned_frames=len(selected),
                reference_points=sum(s["reference_points"] for s in stats),
                complete_profiles=sum(s["complete_profiles"] for s in stats),
                frames_with_profiles=sum(s["complete_profiles"] > 0 for s in stats),
            )
            for axis in ("x", "y"):
                available = [s[axis] for s in stats if s[axis]["points"]]
                unique = [s for s in available if s["unique_peak_points"]]
                record[axis] = {
                    **{
                        k: sum(s[axis][k] for s in stats)
                        for k in ("unique_peak_points", "tied_peak_points", "boundary_peak_points")
                    },
                    "offset_counts": {
                        str(v): sum(s[axis]["offset_counts"][str(v)] for s in stats)
                        for v in OFFSETS
                    },
                    "median_of_frame_median_unique_offset_px": float(
                        np.median([s["median_unique_offset_px"] for s in unique])
                    )
                    if unique
                    else None,
                    "median_of_frame_median_gain": float(
                        np.median([s["median_peak_gain"] for s in available])
                    )
                    if available
                    else None,
                    "median_of_frame_median_separated_gap": float(
                        np.median([s["median_separated_peak_gap"] for s in unique])
                    )
                    if unique
                    else None,
                }
            result.append(record)
    return result
