"""Independent annotation structure checks; never a physical qualification gate."""

from __future__ import annotations

import math


def review_template(manifest: dict, manifest_sha256: str, slot: str) -> dict:
    return {
        "schema": "aero_physical_review_v1",
        "manifest_sha256": manifest_sha256,
        "reviewer_slot": slot,
        "reviewer_identity": "",
        "independent_model_blind_attestation": False,
        "pairs": [
            {"pair_id": p["pair_id"], "reviewed": False, "notes": "", "landmarks": []}
            for p in manifest["pairs"]
        ],
    }


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_reviews(manifest: dict, manifest_sha256: str, reviews: list[dict]) -> dict:
    """Validate annotations, retaining disagreements rather than adjudicating them."""
    errors, identities, slots = [], [], []
    expected = {p["pair_id"]: p for p in manifest["pairs"]}
    if len(expected) != len(manifest["pairs"]):
        errors.append("duplicate manifest pair IDs")
    if manifest.get("split") != "train":
        errors.append("this development panel must be train-only")
    if len(reviews) != 2:
        errors.append("exactly two independent reviews required")
    validated = []
    for index, review in enumerate(reviews):
        prefix = f"review {index}"
        if not isinstance(review, dict):
            errors.append(f"{prefix}: review must be an object")
            validated.append({})
            continue
        identity = review.get("reviewer_identity", "")
        identity = identity.strip() if isinstance(identity, str) else ""
        identities.append(identity.casefold())
        slot = review.get("reviewer_slot")
        slots.append(slot if isinstance(slot, str) else "invalid")
        if not identity:
            errors.append(f"{prefix}: reviewer identity missing")
        if review.get("schema") != "aero_physical_review_v1":
            errors.append(f"{prefix}: unknown schema")
        if review.get("manifest_sha256") != manifest_sha256:
            errors.append(f"{prefix}: manifest hash mismatch")
        if review.get("independent_model_blind_attestation") is not True:
            errors.append(f"{prefix}: independent model-blind attestation missing")
        entries = review.get("pairs", [])
        if not isinstance(entries, list) or any(not isinstance(p, dict) for p in entries):
            errors.append(f"{prefix}: pairs must be objects")
            validated.append({})
            continue
        keys = [p.get("pair_id") for p in entries]
        if any(not isinstance(k, str) for k in keys):
            errors.append(f"{prefix}: pair IDs must be strings")
            validated.append({})
            continue
        if len(set(keys)) != len(keys) or set(keys) != set(expected):
            errors.append(f"{prefix}: incomplete, duplicate or unexpected pairs")
        accepted = {}
        for pair in entries:
            pair_id = pair.get("pair_id")
            if pair_id not in expected:
                continue
            if pair.get("reviewed") is not True:
                errors.append(f"{prefix}/{pair_id}: not reviewed")
            landmarks = pair.get("landmarks", [])
            if not isinstance(landmarks, list):
                errors.append(f"{prefix}/{pair_id}: landmarks must be a list")
                continue
            if not landmarks and not str(pair.get("notes", "")).strip():
                errors.append(f"{prefix}/{pair_id}: no landmarks or observability explanation")
            observed = {}
            for landmark in landmarks:
                if not isinstance(landmark, dict):
                    errors.append(f"{prefix}/{pair_id}: malformed landmark")
                    continue
                name = landmark.get("landmark_id")
                if not isinstance(name, str) or not name.strip() or name in observed:
                    errors.append(f"{prefix}/{pair_id}: empty or duplicate landmark ID")
                    continue
                observed[name] = landmark
                if landmark.get("region") not in ("target", "background"):
                    errors.append(f"{prefix}/{pair_id}/{name}: invalid region")
                for modality in ("visible", "infrared"):
                    point = landmark.get(modality, {})
                    if not isinstance(point, dict):
                        errors.append(f"{prefix}/{pair_id}/{name}: malformed {modality}")
                        continue
                    if point.get("visibility") == "unobservable":
                        if point.get("xy") is not None or not str(point.get("reason", "")).strip():
                            errors.append(
                                f"{prefix}/{pair_id}/{name}: unobservable needs null xy/reason"
                            )
                        continue
                    xy, uncertainty = point.get("xy"), point.get("uncertainty_px")
                    h, w = expected[pair_id]["images"][modality]["shape"][:2]
                    valid = (
                        point.get("visibility") == "visible"
                        and isinstance(xy, list)
                        and len(xy) == 2
                        and all(_number(v) for v in xy)
                        and 0 <= xy[0] <= w - 1
                        and 0 <= xy[1] <= h - 1
                        and _number(uncertainty)
                        and uncertainty > 0
                    )
                    if not valid:
                        errors.append(f"{prefix}/{pair_id}/{name}: invalid {modality} point")
            accepted[pair_id] = observed
        validated.append(accepted)
    if len(set(identities)) != len(identities):
        errors.append("reviewer identities must differ")
    if slots != ["A", "B"]:
        errors.append("reviewer slots must be A then B in input order")
    discrepancies = []
    if not errors:
        for pair_id in expected:
            a, b = (r[pair_id] for r in validated)
            for name in sorted(set(a) | set(b)):
                row = {
                    "pair_id": pair_id,
                    "landmark_id": name,
                    "review_A": a.get(name),
                    "review_B": b.get(name),
                    "requires_semantic_adjudication": True,
                }
                if name in a and name in b:
                    row["native_pixel_disagreement"] = {
                        m: math.dist(a[name][m]["xy"], b[name][m]["xy"])
                        if all(r[name][m]["visibility"] == "visible" for r in (a, b))
                        else None
                        for m in ("visible", "infrared")
                    }
                discrepancies.append(row)
    return {
        "structure_valid": not errors,
        "errors": errors,
        "reviewer_discrepancies": discrepancies,
        "independent_correspondence": "hold_pending_adjudication_and_frozen_protocol",
        "generator_training_eligible": "hold",
        "note": "Structure validity is not physical accuracy, coverage or reviewer verification.",
    }
