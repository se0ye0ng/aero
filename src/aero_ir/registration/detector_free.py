"""Pretrained matching adapters and diagnostics; consistency is NOT pixel GT."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


def guard_empty_xoftr_fine(matcher):
    """Suppress upstream's confidence=1 placeholder when all fine scores fail.

    Valid-match behavior and checkpoint tensors are unchanged. The official code
    sets mask[0,0,0]=1 and confidence[0,0,0]=1 in this branch; that is not evidence.
    """
    module = matcher.fine_matching
    original = module.get_fine_sub_match

    def guarded(self, confidence, feat0, feat1, data):
        if not (confidence > self.fine_thr).any():
            self.suppressed_placeholder = True
            return {
                "m_bids": torch.empty(0, dtype=torch.long, device=confidence.device),
                "mkpts0_f": confidence.new_empty((0, 2)),
                "mkpts1_f": confidence.new_empty((0, 2)),
                "mconf_f": confidence.new_empty((0,)),
            }
        return original(confidence, feat0, feat1, data)

    module.get_fine_sub_match = types.MethodType(guarded, module)
    module.suppressed_placeholder = False


def load_matcher(name, weights: Path, vendor: Path, device):
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    if name == "loftr":
        from kornia.feature import LoFTR

        model = LoFTR(pretrained=None)
    elif name == "xoftr":
        namespace = "_aero_xoftr_vendor"
        if namespace not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                namespace,
                vendor / "src/__init__.py",
                submodule_search_locations=[str(vendor / "src")],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            spec.loader.exec_module(module)
        config_module = importlib.import_module(f"{namespace}.config.default")
        xoftr = importlib.import_module(f"{namespace}.xoftr").XoFTR

        def lower(node):
            return (
                {str(k).lower(): lower(v) for k, v in node.items()}
                if hasattr(node, "items")
                else node
            )

        config = lower(config_module.get_cfg_defaults(inference=True))
        model = xoftr(config["xoftr"])
        guard_empty_xoftr_fine(model)
    else:
        raise ValueError("unknown matcher")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().to(device)


def infer(model, name, gray0, gray1, device):
    data = {
        "image0": torch.from_numpy(gray0.copy())[None, None].float().to(device) / 255,
        "image1": torch.from_numpy(gray1.copy())[None, None].float().to(device) / 255,
    }
    if name == "xoftr":
        model.fine_matching.suppressed_placeholder = False
        model(data)
        p0, p1, conf = data["mkpts0_f"], data["mkpts1_f"], data["mconf_f"]
        suppressed = model.fine_matching.suppressed_placeholder
    else:
        output = model(data)
        p0, p1, conf = output["keypoints0"], output["keypoints1"], output["confidence"]
        suppressed = False
    p0, p1, conf = [v.detach().cpu().numpy() for v in (p0, p1, conf)]
    keep = np.isfinite(p0).all(1) & np.isfinite(p1).all(1) & np.isfinite(conf)
    for points, shape in ((p0, gray0.shape), (p1, gray1.shape)):
        keep &= (points >= 0).all(1) & (points <= np.array([shape[1] - 1, shape[0] - 1])).all(1)
    return (
        p0[keep],
        p1[keep],
        conf[keep],
        {
            "raw_count": len(p0),
            "invalid_or_out_of_bounds": int((~keep).sum()),
            "suppressed_fake_fine_match": bool(suppressed),
        },
    )


def inside_box(points, box):
    """Box is xyxy in the SAME resized pixel-coordinate system as points."""
    return ((points >= box[:2]) & (points <= box[2:])).all(1)


def reciprocal_mask(p0, p1, reverse0, reverse1, tolerance=2.0):
    if not len(p0) or not len(reverse0):
        return np.zeros(len(p0), dtype=bool)
    tree = cKDTree(np.concatenate((reverse1, reverse0), axis=1))
    pairs = np.concatenate((p0, p1), axis=1)
    nearby = tree.query_ball_point(pairs, tolerance * np.sqrt(2))
    return np.array(
        [
            any(
                np.linalg.norm(reverse1[j] - a) <= tolerance
                and np.linalg.norm(reverse0[j] - b) <= tolerance
                for j in indices
            )
            for a, b, indices in zip(p0, p1, nearby, strict=True)
        ],
        dtype=bool,
    )


def homography_diagnostic(p0, p1):
    """Random half-fit/half-check, annotation-free; no GT-accuracy interpretation."""
    if len(p0) < 12:
        return {"status": "insufficient_matches", "fit_count": 0, "check_count": 0}
    order = np.random.default_rng(0).permutation(len(p0))
    fit, check = order[::2], order[1::2]
    cv2.setRNGSeed(0)
    matrix, mask = cv2.findHomography(p0[fit], p1[fit], cv2.RANSAC, 3.0, maxIters=2000)
    if matrix is None or not np.isfinite(matrix).all():
        return {"status": "fit_failed", "fit_count": len(fit), "check_count": len(check)}
    projected = cv2.perspectiveTransform(p0[check].astype(np.float64)[None], matrix)[0]
    errors = np.linalg.norm(projected - p1[check], axis=1)
    return {
        "status": "fit",
        "fit_count": len(fit),
        "check_count": len(check),
        "fit_inlier_fraction": float(mask.mean()),
        "check_fraction_within_3px": float((errors <= 3).mean()),
        "check_median_reprojection_px": float(np.median(errors)),
        "matrix": matrix.tolist(),
    }


def pair_metrics(p0, p1, reverse0, reverse1, box0, box1):
    a, b = inside_box(p0, box0), inside_box(p1, box1)
    reciprocal = reciprocal_mask(p0, p1, reverse0, reverse1)
    return {
        "matches": len(p0),
        "source_box_matches": int(a.sum()),
        "target_box_matches": int(b.sum()),
        "both_box_matches": int((a & b).sum()),
        "one_box_only_matches": int((a ^ b).sum()),
        "box_routing_fraction_not_precision": float((a & b).sum() / a.sum()) if a.any() else None,
        "reciprocal_matches": int(reciprocal.sum()),
        "reciprocal_fraction": float(reciprocal.mean()) if len(p0) else None,
        "both_box_reciprocal_matches": int((a & b & reciprocal).sum()),
        "homography_consistency_not_accuracy": homography_diagnostic(p0, p1),
    }, reciprocal


def header_diagnostic(p0, p1, height0, height1, fraction=0.2):
    """Conservative post-hoc header screen, NOT a complete HUD segmentation.

    Based on visible timestamp/text in the initial review. Scene points can also
    occupy this band; central reticles are NOT removed. This does not rerun the
    matcher without HUD, so attention may already have used overlay content.
    """
    excluded = (p0[:, 1] < fraction * height0) | (p1[:, 1] < fraction * height1)
    return {
        "header_either_matches": int(excluded.sum()),
        "header_either_fraction": float(excluded.mean()) if len(p0) else None,
        "non_header_matches": int((~excluded).sum()),
        "non_header_homography_consistency_not_accuracy": homography_diagnostic(
            p0[~excluded], p1[~excluded]
        ),
    }
