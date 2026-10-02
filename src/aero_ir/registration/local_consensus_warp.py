"""Local-consensus TPS with explicit control-selection policies.

The grid path reproduces the preserved exploratory implementation. FPS changes
only control selection; it does not weaken support, consensus or qualification.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, QhullError, cKDTree


def farthest_controls(points, limit):
    """Input order is descending confidence; ties resolve by that frozen order."""
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or type(limit) is not int or limit < 1):
        raise ValueError("finite Nx2 points and positive integer budget required")
    if len(points) <= limit:
        return np.arange(len(points))
    selected = [0]
    distance = np.sum((points-points[0])**2, axis=1)
    distance[0] = -np.inf
    for _ in range(limit-1):
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.sum((points-points[index])**2, axis=1))
        distance[selected] = -np.inf
    return np.array(selected)


def select_controls(rgb, size, accepted, config, policy):
    if policy == "fps":
        return accepted[farthest_controls(rgb[accepted], config["maximum_controls"])].tolist()
    if policy != "grid":
        raise ValueError("unknown control policy")
    cells = np.floor((rgb+.5)/np.asarray(size)*config["control_grid"]).astype(int)
    controls, occupied = [], set()
    for i in accepted:
        key = tuple(cells[i])
        if key not in occupied:
            controls.append(i)
            occupied.add(key)
    return controls


def fit_image_warp(rgb, thermal, confidence, rgb_size, thermal_size, config, policy="grid"):
    rgb, thermal, confidence = [np.asarray(x, dtype=np.float64) for x in
                                (rgb, thermal, confidence)]
    if (rgb.ndim != 2 or rgb.shape[1] != 2 or rgb.shape != thermal.shape
            or confidence.shape != (len(rgb),) or not all(
                np.isfinite(x).all() for x in (rgb, thermal, confidence))):
        raise ValueError("invalid image-only match arrays")
    if not 0 < config["trim_fraction"] <= 1:
        raise ValueError("invalid trim fraction")
    # Highest-confidence duplicate wins; reference coordinates play no role.
    order = np.argsort(-confidence, kind="stable")
    _, unique = np.unique(rgb[order], axis=0, return_index=True)
    chosen = order[np.sort(unique)]
    rgb, thermal, confidence = rgb[chosen], thermal[chosen], confidence[chosen]
    info = dict(input_matches=len(order), unique_rgb_matches=len(rgb))
    if len(rgb) < config["minimum_controls"]:
        return None, dict(info, status="insufficient_matches")
    scale_rgb, scale_t = float(max(rgb_size)), float(max(thermal_size))
    x, y = (rgb+.5)/scale_rgb, (thermal+.5)/scale_t
    k = min(config["neighbors"], len(x)-1)
    if k < 4:
        return None, dict(info, status="insufficient_neighbors")
    tree = cKDTree(x)
    neighbors = tree.query(x, k=k+1)[1]
    residual = np.full(len(x), np.inf)
    for i in range(len(x)):
        ids = neighbors[i][neighbors[i] != i][:k]
        design = np.column_stack((x[ids]-x[i], np.ones(len(ids))))
        target = y[ids]
        selected = np.arange(len(ids))
        # Point i is excluded: a false match cannot validate itself by interpolation.
        for _ in range(config["trim_iterations"]):
            coef, _, rank, _ = np.linalg.lstsq(design[selected], target[selected], rcond=None)
            if rank < 3:
                break
            errors = np.linalg.norm(design@coef-target, axis=1)
            count = max(4, int(np.ceil(config["trim_fraction"]*len(ids))))
            selected = np.argsort(errors, kind="stable")[:count]
        else:
            coef, _, rank, _ = np.linalg.lstsq(design[selected], target[selected], rcond=None)
            if rank == 3:
                residual[i] = np.linalg.norm(coef[-1]-y[i])*scale_t
    threshold = max(config["agreement_min_thermal_pixels"],
                    config["agreement_thermal_long_side_fraction"]*scale_t)
    accepted = np.flatnonzero(residual <= threshold)
    controls = select_controls(rgb, rgb_size, accepted, config, policy)
    info.update(accepted_matches=len(accepted), controls=len(controls),
                agreement_threshold_thermal_pixels=threshold,
                control_match_indices=chosen[controls].tolist())
    if len(controls) < config["minimum_controls"]:
        return None, dict(info, status="insufficient_controls")
    try:
        fit_x, fit_y = x[controls], y[controls]
        interpolator = RBFInterpolator(fit_x, fit_y, kernel="thin_plate_spline", degree=1,
                                      smoothing=config["tps_smoothing"])
        hull = Delaunay(fit_x)
    except (ValueError, np.linalg.LinAlgError, QhullError):
        return None, dict(info, status="degenerate_controls")

    def predict(query):
        q = (np.asarray(query, dtype=np.float64)+.5)/scale_rgb
        out = np.full_like(q, np.nan)
        supported = hull.find_simplex(q) >= 0
        out[supported] = interpolator(q[supported])*scale_t-.5
        # Unsupported extrapolation and out-of-frame outputs count as failures.
        supported &= np.isfinite(out).all(1) & (out >= 0).all(1)
        supported &= (out <= np.asarray(thermal_size)-1).all(1)
        out[~supported] = np.nan
        return out

    return predict, dict(info, status="fit")
