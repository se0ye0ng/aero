"""Train-only mask diagnostic: forbid trading worse boundaries for better overlap.

This is a local, matched-candidate ablation of saved SAM v2 fits, not a new
qualification gate. Neither the original prompts nor their perturbations are GT.
"""

from itertools import product

import numpy as np

from aero_ir.registration.silhouette_search import matrix_for, objective, score, within_bounds

IDENTITY = (1.0, 0.0, 0.0, 0.0)
METRICS = ("dice_loss", "chamfer_model_px")


def non_deteriorating(value, baseline):
    if value is None or baseline is None:
        return False
    return all(
        np.isfinite(value[k]) and np.isfinite(baseline[k]) and value[k] <= baseline[k] + 1e-10
        for k in METRICS
    )


def candidates_around(parameters):
    """Identity, saved fit, and one fixed 3^4 neighbourhood; no label-dependent budget."""
    p = tuple(float(parameters[k]) for k in ("scale", "angle_degrees", "dx", "dy"))
    if not np.isfinite(p).all() or not within_bounds(p):
        raise ValueError("saved parameters outside declared v2 bounds")
    s, angle, dx, dy = p
    candidates = {IDENTITY, p}
    for ds, da, xx, yy in product((-1, 0, 1), repeat=4):
        q = (s * 1.05**ds, angle + 3 * da, dx + 2 * xx, dy + 2 * yy)
        if within_bounds(q):
            candidates.add(q)
    return sorted(candidates)


def select(records, *, constrained):
    baseline = next(r["score"] for r in records if tuple(r["parameters"]) == IDENTITY)
    if constrained and baseline is None:
        return None
    valid = [
        r
        for r in records
        if r["score"] is not None and (not constrained or non_deteriorating(r["score"], baseline))
    ]
    return min(
        valid,
        key=lambda r: (
            objective(r["score"], 256),
            sum(abs(a - b) for a, b in zip(r["parameters"], IDENTITY, strict=True)),
            tuple(r["parameters"]),
        ),
        default=None,
    )


def compare(source, target, previous):
    a, b = source["masks"], target["masks"]
    va, vb = ~source["excluded"], ~target["excluded"]
    if a.shape != (3, 256, 256) or b.shape != a.shape:
        raise ValueError("requires three 256px original/perturbed masks per modality")
    if (
        any(x.dtype != np.bool_ for x in (a, b, va, vb))
        or va.shape != a.shape[1:]
        or vb.shape != va.shape
    ):
        raise ValueError("requires boolean masks and same-size boolean exclusion support")
    records = [
        {"parameters": p, "score": score(a[0], b[0], matrix_for(p, 256), va, vb)}
        for p in candidates_around(previous["parameters"])
    ]
    baseline = next(r["score"] for r in records if tuple(r["parameters"]) == IDENTITY)
    result = {"baseline": baseline, "candidate_count": len(records), "candidates": records}
    for name, constrained in (("unconstrained", False), ("non_deteriorating", True)):
        chosen = select(records, constrained=constrained)
        if chosen is None:
            result[name] = {"status": "abstain_no_safe_baseline_or_candidate"}
            continue
        h = matrix_for(chosen["parameters"], 256)
        native = (
            np.array(target["meta"]["crop_to_native"])
            @ h
            @ np.array(source["meta"]["native_to_crop"])
        )
        result[name] = {
            "status": "identity_fallback"
            if tuple(chosen["parameters"]) == IDENTITY
            else "fit_pseudo_masks_only",
            "parameters": chosen["parameters"],
            "score": chosen["score"],
            "matrix": h.tolist(),
            "inverse_matrix": np.linalg.inv(h).tolist(),
            "native_rgb_to_ir": native.tolist(),
            "native_ir_to_rgb": np.linalg.inv(native).tolist(),
            "perturbed_prompt_checks_not_gt": [
                {"baseline": score(x, y, np.eye(3), va, vb), "selected": score(x, y, h, va, vb)}
                for x, y in zip(a[1:], b[1:], strict=True)
            ],
        }
    return result
