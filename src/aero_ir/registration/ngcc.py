"""Explicit gradient-vector similarity diagnostics; not physical match confidence."""

import torch
from torch.nn import functional as F

METHODS = ("signed_ngcc", "absolute_ngcc", "ngf_style_squared_cosine")


def grayscale(image):
    if image.shape[1] == 1:
        return image
    return (image * image.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(
        1, keepdim=True
    )


def gradients(image):
    image = grayscale(image)
    kernel = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8
    return F.conv2d(image, torch.stack((kernel, kernel.T))[:, None], padding=1)


def scores(warped_source, target, support):
    """Compute gradients AFTER warp, thus expressed in target coordinates.

    Returns differentiable losses and eligibility for each candidate. Fixed
    support is caller-owned; no candidate-dependent texture masking is used.
    """
    a = gradients(warped_source)[:, :, support]
    b = gradients(target)[:, :, support]
    ac, bc = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    ea, eb = ac.square().mean((1, 2)), bc.square().mean((1, 2))
    valid = (ea > 1e-8) & (eb > 1e-8) & (support.sum() >= 32)
    corr = (ac * bc).sum((1, 2)) / (ac.square().sum((1, 2)) * bc.square().sum((1, 2))).clamp_min(
        1e-24
    ).sqrt()
    local = (a * b).sum(1).square() / ((a.square().sum(1) + 1e-6) * (b.square().sum(1) + 1e-6))
    return dict(zip(METHODS, (1 - corr, 1 - corr.abs(), 1 - local.mean(1)), strict=True)), valid
