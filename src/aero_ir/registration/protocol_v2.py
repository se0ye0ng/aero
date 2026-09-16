"""Frozen geometry-first Anti-UAV300 dense-registration protocol v2.

This module contains no dataset or validation access. The v1 checkpoint and
validation result are immutable; v2 changes the *training* coverage and loss.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from aero_ir.registration.superfusion import (
    box_perimeter_points,
    warp_source_to_target,
)

EPOCHS = 300
PAIRS_PER_SEQUENCE_PER_EPOCH = 16
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
INITIAL_CHECKPOINT_SHA256 = "a4c8aafe95c098f8b0803980be520aae12122c95f0ed89bddcb5c89305f2388a"
LOSS_WEIGHTS = {
    "box_iou": 1.0,
    "centroid_fraction": 0.25,
    "log_area_ratio": 0.25,
    "worst_perimeter_fraction": 0.5,
    "structural_edge_ncc": 0.05,
    "flow_smoothness": 0.10,
    "flow_fold": 0.10,
    "flow_boundary": 0.10,
}


def rotating_positions(
    length: int,
    epoch: int,
    sequence_id: str,
    *,
    seed: int = 0,
    per_epoch: int = PAIRS_PER_SEQUENCE_PER_EPOCH,
) -> np.ndarray:
    """Cycle through all usable pairs before repeating; reshuffle each cycle.

    The official train split's largest sequence has 1,000 usable pairs. At 16
    pairs per epoch, every pair is seen within 63 epochs, not a fixed 16-frame
    cache repeated 300 times. Selection depends only on train annotations.
    """
    if length <= 0 or epoch < 0 or per_epoch <= 0:
        raise ValueError("length and per_epoch must be positive; epoch must be non-negative")
    stable_id = int.from_bytes(hashlib.sha256(sequence_id.encode()).digest()[:8], "little")
    positions = np.arange(epoch * per_epoch, (epoch + 1) * per_epoch, dtype=np.int64)
    cycles, within_cycles = np.divmod(positions, length)
    result = np.empty(per_epoch, dtype=np.int64)
    for cycle in np.unique(cycles):
        generator = np.random.default_rng(np.random.SeedSequence([seed, stable_id, int(cycle)]))
        permutation = generator.permutation(length)
        mask = cycles == cycle
        result[mask] = permutation[within_cycles[mask]]
    return result


def _edge_map(images: torch.Tensor) -> torch.Tensor:
    grayscale = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    kernel_x = images.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).reshape(
        1, 1, 3, 3
    )
    kernel_y = kernel_x.transpose(-1, -2)
    gradient_x = F.conv2d(grayscale, kernel_x, padding=1)
    gradient_y = F.conv2d(grayscale, kernel_y, padding=1)
    return torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)


def _structural_edge_ncc(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_edges = _edge_map(source)[:, :, 38:243, 13:243].flatten(1)
    target_edges = _edge_map(target)[:, :, 38:243, 13:243].flatten(1)
    source_edges = source_edges - source_edges.mean(dim=1, keepdim=True)
    target_edges = target_edges - target_edges.mean(dim=1, keepdim=True)
    numerator = (source_edges * target_edges).sum(dim=1)
    denominator = torch.linalg.vector_norm(source_edges, dim=1) * torch.linalg.vector_norm(
        target_edges, dim=1
    )
    return numerator / denominator.clamp_min(1e-12)


def _sampling_grid(displacement: torch.Tensor) -> torch.Tensor:
    height, width = displacement.shape[2:]
    vertical = torch.linspace(-1.0, 1.0, height, device=displacement.device)
    horizontal = torch.linspace(-1.0, 1.0, width, device=displacement.device)
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)


def geometry_first_loss(
    visible: torch.Tensor,
    infrared: torch.Tensor,
    source_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    displacement: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize box extents, small-target error and structural correspondence.

    The original v1 mean perimeter objective could be small even when one
    outlying perimeter point changed a transferred box's IoU or area. v2
    backpropagates through the same source-to-target inversion used by audit.
    """
    source_points = box_perimeter_points(source_boxes)
    target_points = box_perimeter_points(target_boxes)
    sampled_flow = (
        F.grid_sample(
            displacement,
            target_points.unsqueeze(2),
            padding_mode="border",
            align_corners=False,
        )
        .squeeze(-1)
        .permute(0, 2, 1)
    )
    predicted_source_points = (target_points + sampled_flow + 1.0) / 2.0
    predicted_min = predicted_source_points.amin(dim=1)
    predicted_max = predicted_source_points.amax(dim=1)
    predicted_boxes = torch.cat(
        ((predicted_min + predicted_max) / 2.0, predicted_max - predicted_min), dim=1
    )
    target_min = source_boxes[:, :2] - source_boxes[:, 2:] / 2.0
    target_max = source_boxes[:, :2] + source_boxes[:, 2:] / 2.0
    overlap_size = (
        torch.minimum(predicted_max, target_max) - torch.maximum(predicted_min, target_min)
    ).clamp_min(0.0)
    overlap_area = overlap_size.prod(dim=1)
    predicted_area = predicted_boxes[:, 2:].prod(dim=1)
    target_area = source_boxes[:, 2:].prod(dim=1)
    iou = overlap_area / (predicted_area + target_area - overlap_area).clamp_min(1e-8)
    box_iou_loss = (1.0 - iou).mean()
    target_diagonal = torch.linalg.vector_norm(source_boxes[:, 2:], dim=1).clamp_min(1e-5)
    centroid_loss = (
        torch.linalg.vector_norm(predicted_boxes[:, :2] - source_boxes[:, :2], dim=1)
        / target_diagonal
    ).mean()
    area_loss = torch.log((predicted_area + 1e-8) / (target_area + 1e-8)).abs().mean()

    max_point_error = torch.linalg.vector_norm(
        target_points + sampled_flow - source_points, dim=-1
    ).amax(dim=1)
    perimeter_loss = (max_point_error / (2.0 * target_diagonal)).mean()

    registered = warp_source_to_target(visible, displacement)
    edge_loss = 1.0 - _structural_edge_ncc(registered, infrared).mean()
    horizontal_smoothness = (displacement[:, :, :, 1:] - displacement[:, :, :, :-1]).abs().mean()
    vertical_smoothness = (displacement[:, :, 1:, :] - displacement[:, :, :-1, :]).abs().mean()
    smoothness_loss = horizontal_smoothness + vertical_smoothness

    height, width = displacement.shape[2:]
    dx = (displacement[:, :, :, 1:] - displacement[:, :, :, :-1]) * ((width - 1) / 2.0)
    dy = (displacement[:, :, 1:, :] - displacement[:, :, :-1, :]) * ((height - 1) / 2.0)
    determinant = (1.0 + dx[:, 0, :-1]) * (1.0 + dy[:, 1, :, :-1]) - (
        dx[:, 1, :-1] * dy[:, 0, :, :-1]
    )
    fold_loss = F.relu(0.2 - determinant).mean()
    boundary_loss = F.relu((_sampling_grid(displacement) + displacement).abs() - 1.0).mean()

    components = {
        "box_iou": box_iou_loss,
        "centroid_fraction": centroid_loss,
        "log_area_ratio": area_loss,
        "worst_perimeter_fraction": perimeter_loss,
        "structural_edge_ncc": edge_loss,
        "flow_smoothness": smoothness_loss,
        "flow_fold": fold_loss,
        "flow_boundary": boundary_loss,
    }
    total = sum(LOSS_WEIGHTS[key] * value for key, value in components.items())
    telemetry = {key: float(value.detach()) for key, value in components.items()}
    telemetry["total"] = float(total.detach())
    telemetry["mean_box_iou"] = float(iou.detach().mean())
    return total, telemetry
