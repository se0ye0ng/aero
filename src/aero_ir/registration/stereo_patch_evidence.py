"""Same-modality patch evidence; not independent correspondence ground truth."""

from __future__ import annotations

import numpy as np

NAMES = ("lidar", "sgbm", "raft")


def patches(image, support, centres, radius=4):
    """Bilinear patches; require every four-neighbour sample to be observed."""
    if (
        image.ndim != 2
        or image.shape != support.shape
        or support.dtype != bool
        or centres.ndim != 2
        or centres.shape[1] != 2
        or radius != 4
        or min(image.shape) < 2
        or not np.isfinite(image).all()
    ):
        raise ValueError("finite grayscale image, boolean support and Nx2 centres required")
    dy, dx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    xy = centres[:, None, :] + np.c_[dx.ravel(), dy.ravel()][None]
    finite = np.isfinite(xy).all(axis=2)
    safe = np.where(finite[..., None], xy, 0)
    x, y = np.floor(safe).astype(np.int64).transpose(2, 0, 1)
    h, w = image.shape
    inside = finite & (x >= 0) & (y >= 0) & (x + 1 < w) & (y + 1 < h)
    xc, yc = np.clip(x, 0, w - 2), np.clip(y, 0, h - 2)
    inside &= support[yc, xc] & support[yc, xc + 1]
    inside &= support[yc + 1, xc] & support[yc + 1, xc + 1]
    ax, ay = (safe - np.stack((x, y), axis=-1)).transpose(2, 0, 1)
    values = (
        (1 - ax) * (1 - ay) * image[yc, xc]
        + ax * (1 - ay) * image[yc, xc + 1]
        + (1 - ax) * ay * image[yc + 1, xc]
        + ax * ay * image[yc + 1, xc + 1]
    )
    valid = inside.all(axis=1)
    values[~valid] = np.nan
    return values, valid


def patch_evidence(left, right, left_support, right_support, xy, disparities):
    """Score fixed centre disparities, without optimizing a warp or filtering by score.

    Each 9x9 patch assumes a constant disparity. Occlusion, depth variation,
    repeated texture and image noise can defeat this objective. A higher NCC is
    stronger local image evidence, not proof of the same physical point.
    """
    if set(disparities) != set(NAMES) or any(d.shape != (len(xy),) for d in disparities.values()):
        raise ValueError("aligned lidar, sgbm and raft disparities required")
    ncc = {name: np.full(len(xy), np.nan) for name in NAMES}
    std = np.full(len(xy), np.nan)
    support = {name: np.zeros(len(xy), bool) for name in NAMES}
    for start in range(0, len(xy), 1024):
        sl = slice(start, start + 1024)
        lp, lv = patches(left, left_support, xy[sl])
        lc = lp - lp.mean(axis=1, keepdims=True)
        ll = np.sum(lc * lc, axis=1)
        std[sl] = np.sqrt(ll / lp.shape[1])
        for name, disparity in disparities.items():
            centre = xy[sl].copy()
            centre[:, 0] -= disparity[sl]
            rp, rv = patches(right, right_support, centre)
            rc = rp - rp.mean(axis=1, keepdims=True)
            rr = np.sum(rc * rc, axis=1)
            valid = lv & rv & np.isfinite(disparity[sl]) & (disparity[sl] > 0)
            support[name][sl] = valid
            textured = valid & (ll > 1e-12) & (rr > 1e-12)
            result = np.full(len(lp), np.nan)
            result[textured] = np.clip(
                np.sum(lc[textured] * rc[textured], axis=1) / np.sqrt(ll[textured] * rr[textured]),
                -1.0,
                1.0,
            )
            ncc[name][sl] = result
    return dict(
        left_std_dn=std,
        **{f"ncc_{k}": v for k, v in ncc.items()},
        **{f"patch_supported_{k}": v for k, v in support.items()},
    )


def summarize(evidence):
    n = len(evidence["left_std_dn"])
    std = evidence["left_std_dn"]
    masks = dict(
        all=np.ones(n, bool),
        left_std_below_2=std < 2,
        left_std_2_to_8=(std >= 2) & (std < 8),
        left_std_at_least_8=std >= 8,
        left_patch_unavailable=~np.isfinite(std),
    )
    result = {}
    for key, mask in masks.items():
        common = mask.copy()
        for name in NAMES:
            common &= np.isfinite(evidence[f"ncc_{name}"])
        stats = dict(reference_points=int(mask.sum()), common_points=int(common.sum()))
        for name in NAMES:
            values = evidence[f"ncc_{name}"]
            stats[name] = dict(
                patch_supported=int((mask & evidence[f"patch_supported_{name}"]).sum()),
                finite_ncc=int((mask & np.isfinite(values)).sum()),
                common_median_ncc=float(np.median(values[common])) if common.any() else None,
            )
        for a, b in (("sgbm", "lidar"), ("raft", "lidar"), ("raft", "sgbm")):
            delta = evidence[f"ncc_{a}"][common] - evidence[f"ncc_{b}"][common]
            stats[f"{a}_minus_{b}"] = dict(
                median_delta=float(np.median(delta)) if len(delta) else None,
                positive=int((delta > 1e-9).sum()),
                negative=int((delta < -1e-9).sum()),
                tied=int((np.abs(delta) <= 1e-9).sum()),
            )
        result[key] = stats
    return result


def aggregate(rows):
    result = []
    for sensor, condition in sorted({(r["sensor"], r["condition"]) for r in rows}):
        selected = [r for r in rows if (r["sensor"], r["condition"]) == (sensor, condition)]
        for key in selected[0]["strata"]:
            stats = [r["strata"][key] for r in selected]
            shared = [s for s in stats if s["common_points"]]
            record = dict(
                sensor=sensor,
                condition=condition,
                stratum=key,
                planned_frames=len(selected),
                common_frames=len(shared),
                reference_points=sum(s["reference_points"] for s in stats),
                common_points=sum(s["common_points"] for s in stats),
            )
            for name in NAMES:
                record[name] = dict(
                    finite_ncc=sum(s[name]["finite_ncc"] for s in stats),
                    patch_supported=sum(s[name]["patch_supported"] for s in stats),
                    median_of_common_frame_median_ncc=float(
                        np.median([s[name]["common_median_ncc"] for s in shared])
                    )
                    if shared
                    else None,
                )
            for pair in ("sgbm_minus_lidar", "raft_minus_lidar", "raft_minus_sgbm"):
                delta = [s[pair]["median_delta"] for s in shared]
                record[pair] = dict(
                    median_of_frame_median_delta=float(np.median(delta)) if delta else None,
                    positive_frames=sum(v > 1e-9 for v in delta),
                    negative_frames=sum(v < -1e-9 for v in delta),
                    tied_frames=sum(abs(v) <= 1e-9 for v in delta),
                )
            result.append(record)
    return result
