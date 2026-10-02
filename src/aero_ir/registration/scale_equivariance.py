"""Experimental source-scale response loss; never a correspondence certificate.

The teacher is a detached, potentially incorrect original forward map. This
regularizer must accompany original geometric supervision: a map collapsed to
the scale centre can satisfy equivariance without aligning either image.
"""

import torch
from torch.nn import functional as F

from aero_ir.registration.geometry import centre_grid, sampling_map, valid_support


def source_scale_loss(reference, prediction, target_boxes, source_centres, scales):
    """Mean per-observation SmoothL1 source-coordinate error in pixels.

Fields: N,2,H,W, centre-lattice normalized displacements. Centres: N,2 in
[-1,1] coordinates; scales: N,2 positive xy factors. Target boxes: N,4
normalized cxcywh. Boxes define loss support only, not inference inputs.
Reference/known-transform support is fixed for this loss evaluation. Samples
with no measurable target support raise rather than silently yield zero loss.
"""
    if reference.shape != prediction.shape or reference.ndim != 4 or reference.shape[1] != 2:
        raise ValueError("matching N,2,H,W fields required")
    n, _, h, w = reference.shape
    if (target_boxes.shape != (n, 4) or source_centres.shape != (n, 2)
            or scales.shape != (n, 2)):
        raise ValueError("invalid boxes, centres or scale shape")
    if not all(torch.isfinite(t).all() for t in
               (reference, prediction, target_boxes, source_centres, scales)):
        raise ValueError("nonfinite scale-loss input")
    if (scales <= 0).any() or (target_boxes[:, 2:] <= 0).any():
        raise ValueError("positive scales and box extents required")
    with torch.no_grad():
        original = sampling_map(reference.detach())
        centre = source_centres.detach()[:, None, None]
        expected = centre + (original - centre) * scales.detach()[:, None, None]
        support = valid_support(original, h, w) & valid_support(expected, h, w)
        grid = (centre_grid(reference) + 1) / 2
        roi = ((grid - target_boxes[:, None, None, :2]).abs()
               <= target_boxes[:, None, None, 2:] / 2).all(-1)
        mask = support & roi
        count = mask.sum((1, 2))
        if (count == 0).any():
            raise ValueError("no measurable target support; do not count as zero loss")
    error = (sampling_map(prediction) - expected) * reference.new_tensor([w/2, h/2])
    per_pixel = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean(-1)
    loss = ((per_pixel * mask).sum((1, 2)) / count).mean()
    return loss, {"target_pixels": count.tolist(),
                  "roi_support_fraction": (count / roi.sum((1, 2)).clamp_min(1)).tolist()}
