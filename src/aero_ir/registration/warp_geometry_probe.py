"""Sample warp geometry without treating local regularity as correspondence accuracy."""

import numpy as np


def sample_geometry(forward, reverse, source_size, step=16, delta=1.0):
    width, height = source_size
    if min(width, height) < 1 or step < 2 or not np.isfinite(delta) or delta <= 0:
        raise ValueError("positive image dimensions, step>=2 and finite delta>0 required")
    yy, xx = np.mgrid[step // 2 : height : step, step // 2 : width : step]
    xy = np.c_[xx.ravel(), yy.ravel()].astype(float)

    def apply(fn, points):
        return np.full_like(points, np.nan) if fn is None else np.asarray(fn(points), dtype=float)

    target = apply(forward, xy)
    dx = (apply(forward, xy + [delta, 0]) - apply(forward, xy - [delta, 0])) / (2 * delta)
    dy = (apply(forward, xy + [0, delta]) - apply(forward, xy - [0, delta])) / (2 * delta)
    jac = np.stack((dx, dy), axis=-1)
    inside = (xy - delta >= 0).all(1) & (xy + delta <= np.array([width - 1, height - 1])).all(1)
    valid = np.isfinite(jac).all((1, 2)) & np.isfinite(target).all(1) & inside
    determinant, singular = np.full(len(xy), np.nan), np.full((len(xy), 2), np.nan)
    determinant[valid] = np.linalg.det(jac[valid])
    singular[valid] = np.linalg.svd(jac[valid], compute_uv=False)
    back = np.full_like(xy, np.nan)
    available = np.isfinite(target).all(1)
    if reverse is not None and available.any():
        back[available] = apply(reverse, target[available])
    cycle = np.linalg.norm(back - xy, axis=1)
    finite_cycle = np.isfinite(cycle)
    summary = dict(
        grid_points=len(xy),
        forward_supported=int(available.sum()),
        jacobian_supported=int(valid.sum()),
        sampled_nonpositive_determinants=int(np.count_nonzero(determinant[valid] <= 0)),
        sampled_min_determinant=float(np.min(determinant[valid])) if valid.any() else None,
        sampled_min_singular_value=float(np.min(singular[valid, 1])) if valid.any() else None,
        cycle_supported=int(finite_cycle.sum()),
        cycle_median_native_source_px=float(np.median(cycle[finite_cycle]))
        if finite_cycle.any()
        else None,
        cycle_p95_native_source_px=float(np.percentile(cycle[finite_cycle], 95))
        if finite_cycle.any()
        else None,
        fraction_all_grid_cycle_within_3px=float(np.count_nonzero(cycle <= 3) / len(xy))
        if len(xy)
        else None,
    )
    return dict(
        source_xy=xy,
        target_xy=target,
        jacobian=jac,
        jacobian_supported=valid,
        determinant=determinant,
        singular_values=singular,
        backward_xy=back,
        cycle_error=cycle,
    ), summary
