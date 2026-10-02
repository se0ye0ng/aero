"""Vertical image evidence conditional on fixed horizontal disparity anchors."""

from __future__ import annotations

import numpy as np

from aero_ir.registration.stereo_patch_evidence import NAMES, patches
from aero_ir.registration.stereo_patch_profile import OFFSETS, peak_statistics


def vertical_profile(left, right, left_support, right_support, xy, disparity):
    if disparity.shape != (len(xy),):
        raise ValueError("one fixed disparity per point required")
    scores = np.full((len(xy), len(OFFSETS)), np.nan)
    for start in range(0, len(xy), 1024):
        sl = slice(start, start + 1024)
        lp, lv = patches(left, left_support, xy[sl])
        lc = lp - lp.mean(axis=1, keepdims=True)
        ll = (lc * lc).sum(axis=1)
        origin = xy[sl].copy()
        origin[:, 0] -= disparity[sl]
        for j, offset in enumerate(OFFSETS):
            shifted = origin.copy()
            shifted[:, 1] += offset
            rp, rv = patches(right, right_support, shifted)
            rc = rp - rp.mean(axis=1, keepdims=True)
            rr = (rc * rc).sum(axis=1)
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
                (lc[valid] * rc[valid]).sum(axis=1) / np.sqrt(ll[valid] * rr[valid]), -1.0, 1.0
            )
            scores[sl, j] = values
    return scores


def known_shift_control(image, support, xy, vertical_shift):
    """Real-texture execution control with exact integer shifts, not a data qualifier."""
    if vertical_shift not in (0, 1):
        raise ValueError("fixed zero/one vertical controls required")
    h, w = image.shape
    right, mask = np.zeros_like(image), np.zeros_like(support)
    right[vertical_shift:, : w - 6] = image[: h - vertical_shift, 6:]
    mask[vertical_shift:, : w - 6] = support[: h - vertical_shift, 6:]
    scores = vertical_profile(image, right, support, mask, xy, np.full(len(xy), 6.0))
    complete = np.isfinite(scores).all(axis=1)
    expected = np.flatnonzero(OFFSETS == vertical_shift)[0]
    ok = bool(
        complete.any() and np.all(scores[complete, expected] >= scores[complete].max(axis=1) - 1e-9)
    )
    return scores, dict(
        expected_y_offset_px=vertical_shift,
        sampled_points=len(xy),
        complete_profiles=int(complete.sum()),
        ok=ok,
        peaks=peak_statistics(scores, complete),
    )


def summarize(profiles, std):
    if set(profiles) != set(NAMES) or any(
        v.shape != (len(std), len(OFFSETS)) for v in profiles.values()
    ):
        raise ValueError("three aligned anchor profiles required")
    available = {name: np.isfinite(v).all(axis=1) for name, v in profiles.items()}
    common = np.logical_and.reduce(list(available.values()))
    masks = dict(
        all=np.ones(len(std), bool),
        left_std_below_2=std < 2,
        left_std_2_to_8=(std >= 2) & (std < 8),
        left_std_at_least_8=std >= 8,
        left_patch_unavailable=~np.isfinite(std),
    )
    return {
        key: dict(
            reference_points=int(mask.sum()),
            common_points=int((mask & common).sum()),
            anchors={
                name: dict(
                    available_profiles=int((mask & available[name]).sum()),
                    common=peak_statistics(values, mask & common),
                )
                for name, values in profiles.items()
            },
        )
        for key, mask in masks.items()
    }


def aggregate(rows):
    results = []
    for sensor, condition in sorted({(r["sensor"], r["condition"]) for r in rows}):
        selected = [r for r in rows if (r["sensor"], r["condition"]) == (sensor, condition)]
        for key in selected[0]["strata"]:
            stats = [r["strata"][key] for r in selected]
            record = dict(
                sensor=sensor,
                condition=condition,
                stratum=key,
                planned_frames=len(selected),
                reference_points=sum(s["reference_points"] for s in stats),
                common_points=sum(s["common_points"] for s in stats),
                common_frames=sum(s["common_points"] > 0 for s in stats),
                anchors={},
            )
            for name in NAMES:
                shared = [s["anchors"][name]["common"] for s in stats if s["common_points"]]
                unique = [s for s in shared if s["unique_peak_points"]]
                record["anchors"][name] = dict(
                    available_profiles=sum(s["anchors"][name]["available_profiles"] for s in stats),
                    unique_peak_points=sum(s["unique_peak_points"] for s in shared),
                    tied_peak_points=sum(s["tied_peak_points"] for s in shared),
                    boundary_peak_points=sum(s["boundary_peak_points"] for s in shared),
                    median_of_common_frame_peak_offsets=float(
                        np.median([s["median_unique_offset_px"] for s in unique])
                    )
                    if unique
                    else None,
                    median_of_common_frame_peak_gain=float(
                        np.median([s["median_peak_gain"] for s in shared])
                    )
                    if shared
                    else None,
                    median_of_common_frame_separated_gap=float(
                        np.median([s["median_separated_peak_gap"] for s in unique])
                    )
                    if unique
                    else None,
                    offset_counts={
                        str(v): sum(s["offset_counts"][str(v)] for s in shared) for v in OFFSETS
                    },
                )
            results.append(record)
    return results
