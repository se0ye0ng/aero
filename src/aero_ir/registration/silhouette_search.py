"""Bounded multiresolution similarity search; pseudo-mask fit is not physical GT."""

from itertools import product

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from aero_ir.registration.roi_geometry import warp


def matrix_for(parameters, side):
    scale, angle, dx, dy = parameters
    if scale <= 0 or not np.isfinite(parameters).all():
        raise ValueError("invalid similarity parameters")
    matrix = cv2.getRotationMatrix2D(((side - 1) / 2,) * 2, angle, scale)
    matrix[:, 2] += [dx, dy]
    return np.vstack((matrix, [0.0, 0.0, 1.0]))


def score(a, b, matrix, valid_a, valid_b):
    """Symmetric Dice + Euclidean boundary distance, never score missing foreground."""
    results = []
    for source, target, source_valid, target_valid, h in (
        (a, b, valid_a, valid_b, matrix),
        (b, a, valid_b, valid_a, np.linalg.inv(matrix)),
    ):
        side = target.shape[0]
        warped = warp(source.astype(np.uint8), h, side, nearest=True).astype(bool)
        support = warp(source_valid.astype(np.float32), h, side) >= 1 - 1e-6
        support &= target_valid
        safe = binary_erosion(support, structure=np.ones((3, 3)), border_value=0)
        if not warped.any() or not target.any() or (warped & ~safe).any() or (target & ~safe).any():
            return None
        ea = warped & ~binary_erosion(warped)
        eb = target & ~binary_erosion(target)
        dice = 1 - 2 * (warped & target).sum() / (warped.sum() + target.sum())
        chamfer = (
            distance_transform_edt(~ea)[eb].mean() + distance_transform_edt(~eb)[ea].mean()
        ) / 2
        results.append([dice, chamfer])
    dice, chamfer = np.mean(results, axis=0)
    return {"dice_loss": float(dice), "chamfer_model_px": float(chamfer)}


def objective(value, side):
    return float("inf") if value is None else value["dice_loss"] + value["chamfer_model_px"] / side


def within_bounds(p):
    scale, angle, dx, dy = p
    return 0.5 <= scale <= 2.0 and abs(angle) <= 45 and max(abs(dx), abs(dy)) <= 64


def search(a, b, valid_a, valid_b):
    side = a.shape[0]
    if (
        a.shape != b.shape
        or a.shape != (side, side)
        or side != 256
        or valid_a.shape != a.shape
        or valid_b.shape != b.shape
    ):
        raise ValueError("search requires paired 256x256 masks/support")
    if not a.any() or not b.any():
        return {"status": "no_foreground_safe_transform", "evaluated_candidates": {}}
    identity = (1.0, 0.0, 0.0, 0.0)
    counts = {}

    def scorer(size):
        fraction = size / side
        aa, bb = [
            cv2.resize(
                x.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST_EXACT
            ).astype(bool)
            for x in (a, b)
        ]
        va, vb = [
            cv2.resize(x.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) >= 1 - 1e-6
            for x in (valid_a, valid_b)
        ]
        # Pixel-centre resize conjugation (not scaling translations only).
        resize = np.array(
            [[fraction, 0, (fraction - 1) / 2], [0, fraction, (fraction - 1) / 2], [0, 0, 1.0]]
        )
        cache = {}

        def evaluate(p):
            p = tuple(float(x) for x in p)
            if p not in cache:
                h = resize @ matrix_for(p, side) @ np.linalg.inv(resize)
                cache[p] = score(aa, bb, h, va, vb)
                counts[str(size)] = counts.get(str(size), 0) + 1
            return objective(cache[p], size), cache[p]

        return evaluate

    coarse = scorer(64)
    grid = list(
        product(
            (0.6, 0.8, 1.0, 1.25, 1.667),
            (-30.0, -15.0, 0.0, 15.0, 30.0),
            (-32.0, -16.0, 0.0, 16.0, 32.0),
            (-32.0, -16.0, 0.0, 16.0, 32.0),
        )
    )
    # 625 deterministic candidates, no boxes, perturbed masks, or pair labels in ranking.
    seeds = sorted(
        grid,
        key=lambda p: (coarse(p)[0], sum(abs(x - y) for x, y in zip(p, identity, strict=True)), p),
    )[:2]
    history = [identity, *seeds]
    for size, scale_step, angle_step, move_step in (
        (128, 1.15, 10.0, 8.0),
        (256, 1.05, 3.0, 2.0),
        (256, 1.02, 1.0, 1.0),
    ):
        evaluate = scorer(size)
        candidates = {identity, *history}
        for s, angle, dx, dy in seeds:
            for ds, da, xx, yy in product((-1, 0, 1), repeat=4):
                p = (
                    s * scale_step**ds,
                    angle + da * angle_step,
                    dx + xx * move_step,
                    dy + yy * move_step,
                )
                if within_bounds(p):
                    candidates.add(p)
        seeds = sorted(
            candidates,
            key=lambda p: (
                evaluate(p)[0],
                sum(abs(x - y) for x, y in zip(p, identity, strict=True)),
                p,
            ),
        )[:2]
        history.extend(seeds)
    final = scorer(side)
    p = min(
        history,
        key=lambda p: (final(p)[0], sum(abs(x - y) for x, y in zip(p, identity, strict=True)), p),
    )
    selected = final(p)[1]
    if selected is None:
        return {"status": "no_foreground_safe_transform", "evaluated_candidates": counts}
    h = matrix_for(p, side)
    return {
        "status": "fit_pseudo_masks_only",
        "baseline": final(identity)[1],
        "selected": selected,
        "parameters": dict(zip(("scale", "angle_degrees", "dx", "dy"), p, strict=True)),
        "matrix": h.tolist(),
        "inverse_matrix": np.linalg.inv(h).tolist(),
        "determinant": float(np.linalg.det(h)),
        "evaluated_candidates": counts,
        "at_hard_search_boundary": p[0] <= 0.50001
        or p[0] >= 1.99999
        or abs(p[1]) >= 44.999
        or max(abs(p[2]), abs(p[3])) >= 63.999,
        "near_explored_translation_edge": max(abs(p[2]), abs(p[3])) >= 40,
        "near_explored_angle_edge": abs(p[1]) >= 40,
        "outside_initial_scale_grid": p[0] < 0.6 or p[0] > 1.667,
        "chamfer_worsened_from_baseline": final(identity)[1] is not None
        and selected["chamfer_model_px"] > final(identity)[1]["chamfer_model_px"] + 1e-10,
    }


def align_masks(source, target):
    a, b = source["masks"], target["masks"]
    va, vb = ~source["excluded"], ~target["excluded"]
    result = search(a[0], b[0], va, vb)
    if result["status"] == "fit_pseudo_masks_only":
        h = np.array(result["matrix"])
        result["perturbed_prompt_checks_not_gt"] = [
            {"baseline": score(x, y, np.eye(3), va, vb), "selected": score(x, y, h, va, vb)}
            for x, y in zip(a[1:], b[1:], strict=True)
        ]
        native = (
            np.array(target["meta"]["crop_to_native"])
            @ h
            @ np.array(source["meta"]["native_to_crop"])
        )
        result["native_rgb_to_ir"] = native.tolist()
        result["native_ir_to_rgb"] = np.linalg.inv(native).tolist()
    return result
