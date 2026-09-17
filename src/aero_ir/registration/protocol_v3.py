"""Pre-registered bidirectional Anti-UAV300 registration protocol v3.

SuperFusion emits backward sampling maps: a visible-to-infrared warp maps
infrared output coordinates to visible input coordinates.  Protocol v3 scores
that map in its native direction and obtains the opposite map from the
matcher's native reverse direction.  It never treats inversion of an
axis-aligned box enclosure as an image-registration measurement.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from aero_ir.registration.protocol_v2 import geometry_first_loss

EPOCHS = 300
PAIRS_PER_SEQUENCE_PER_EPOCH = 16
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
INITIAL_CHECKPOINT_SHA256 = "13dd5c4700aa072e8b51197222d0d36675e4ab5a4879abd940576d579331135f"
LOSS_WEIGHTS = {
    "visible_to_infrared_geometry": 0.5,
    "infrared_to_visible_geometry": 0.5,
    "inverse_consistency": 0.25,
}


def _grid(displacement: torch.Tensor) -> torch.Tensor:
    height, width = displacement.shape[2:]
    vertical = torch.linspace(
        -1.0, 1.0, height, device=displacement.device, dtype=displacement.dtype
    )
    horizontal = torch.linspace(
        -1.0, 1.0, width, device=displacement.device, dtype=displacement.dtype
    )
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def compose_backward_displacements(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose reciprocal backward fields and return residual and valid mask.

    ``first`` maps coordinates in domain A to domain B. ``second`` maps B to
    A.  The returned residual is ``second(first(a)) - a`` in normalized-grid
    units.  The validity mask excludes compositions that leave domain B.
    """
    if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 2:
        raise ValueError("reciprocal displacements must have equal N,2,H,W shapes")
    base = _grid(first)
    intermediate = base + first.permute(0, 2, 3, 1)
    sampled_second = F.grid_sample(
        second,
        intermediate,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    residual = first + sampled_second
    valid = ((intermediate >= -1.0) & (intermediate <= 1.0)).all(dim=-1)
    return residual, valid


def inverse_consistency_statistics(
    visible_to_infrared: torch.Tensor,
    infrared_to_visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-image cycle residuals for both native sampling directions."""
    target_cycle, target_valid = compose_backward_displacements(
        visible_to_infrared, infrared_to_visible
    )
    source_cycle, source_valid = compose_backward_displacements(
        infrared_to_visible, visible_to_infrared
    )

    def reduce(residual: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = torch.linalg.vector_norm(residual, dim=1)
        masked = norm.masked_fill(~valid, 0.0)
        mean = masked.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
        maximum = norm.masked_fill(~valid, float("-inf")).amax(dim=(1, 2))
        maximum = torch.where(
            valid.any(dim=(1, 2)), maximum, torch.full_like(maximum, float("inf"))
        )
        return mean, maximum

    target_mean, target_max = reduce(target_cycle, target_valid)
    source_mean, source_max = reduce(source_cycle, source_valid)
    return {
        "target_cycle_mean": target_mean,
        "target_cycle_max": target_max,
        "source_cycle_mean": source_mean,
        "source_cycle_max": source_max,
        "target_cycle_valid_fraction": target_valid.float().mean(dim=(1, 2)),
        "source_cycle_valid_fraction": source_valid.float().mean(dim=(1, 2)),
    }


def bidirectional_geometry_loss(
    visible: torch.Tensor,
    infrared: torch.Tensor,
    visible_boxes: torch.Tensor,
    infrared_boxes: torch.Tensor,
    visible_to_infrared: torch.Tensor,
    infrared_to_visible: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train both native backward maps plus their off-grid composition."""
    forward_loss, forward_values = geometry_first_loss(
        visible,
        infrared,
        visible_boxes,
        infrared_boxes,
        visible_to_infrared,
    )
    reverse_loss, reverse_values = geometry_first_loss(
        infrared,
        visible,
        infrared_boxes,
        visible_boxes,
        infrared_to_visible,
    )
    consistency = inverse_consistency_statistics(visible_to_infrared, infrared_to_visible)
    consistency_loss = 0.5 * (
        consistency["target_cycle_mean"].mean() + consistency["source_cycle_mean"].mean()
    )
    total = (
        LOSS_WEIGHTS["visible_to_infrared_geometry"] * forward_loss
        + LOSS_WEIGHTS["infrared_to_visible_geometry"] * reverse_loss
        + LOSS_WEIGHTS["inverse_consistency"] * consistency_loss
    )
    telemetry = {
        **{f"visible_to_infrared_{key}": value for key, value in forward_values.items()},
        **{f"infrared_to_visible_{key}": value for key, value in reverse_values.items()},
        "inverse_consistency": float(consistency_loss.detach()),
        "total": float(total.detach()),
    }
    return total, telemetry
