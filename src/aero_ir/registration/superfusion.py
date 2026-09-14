"""Device-safe SuperFusion dense matcher used for RGB/IR registration.

The network architecture in this module is adapted from the MIT-licensed
SuperFusion implementation by Linfeng Tang. Only the registration network is
included. See ``THIRD_PARTY_NOTICES.md`` for attribution.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    import kornia.filters as KF
except ImportError as error:  # pragma: no cover - checked by the GPU entry point
    raise RuntimeError(
        "SuperFusion registration requires kornia==0.6.5; install the registration extra"
    ) from error


def _create_meshgrid(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Match Kornia 0.6.5's normalized x/y meshgrid without its legacy warning."""
    vertical = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    horizontal = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


class Conv2d(nn.Module):
    """Convolution block retaining upstream parameter names for checkpoint loading."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        norm: type[nn.Module] | None = None,
        act: type[nn.Module] | None = nn.LeakyReLU,
        bias: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                n_in,
                n_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                dilation=dilation,
            )
        ]
        if norm is not None:
            layers.append(norm(n_out, affine=False))
        if act is nn.LeakyReLU:
            layers.append(act(negative_slope=0.1, inplace=True))
        elif act is not None:
            layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class ResConv2d(nn.Module):
    """Residual convolution retaining upstream parameter names."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        dilation: int = 1,
        norm: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                n_in,
                n_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                dilation=dilation,
            )
        ]
        if norm is not None:
            layers.append(norm(n_out, affine=False))
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value) + value


class FeatureExtractorUnshared(nn.Module):
    def __init__(
        self,
        depth: int,
        base_ic: int,
        base_oc: int,
        base_dilation: int,
        norm: type[nn.Module],
    ) -> None:
        super().__init__()
        layers = nn.ModuleList()
        input_channels = base_ic
        output_channels = base_oc
        dilation = base_dilation
        for index in range(depth):
            if index % 2 == 1:
                dilation *= 2
            block = ResConv2d if input_channels == output_channels else Conv2d
            layers.append(
                block(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    norm=norm,
                )
            )
            input_channels = output_channels
            if index % 2 == 1 and index < depth - 1:
                output_channels *= 2
        self.ic = input_channels
        self.oc = output_channels
        self.dilation = dilation
        self.layers = layers

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class DisplacementEstimator(nn.Module):
    def __init__(
        self,
        channel: int,
        depth: int = 4,
        norm: type[nn.Module] = nn.BatchNorm2d,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.corrks = 7
        self.preprocessor = Conv2d(
            channel,
            channel,
            3,
            act=None,
            norm=None,
            dilation=dilation,
            padding=dilation,
        )
        self.featcompressor = nn.Sequential(
            Conv2d(channel * 2, channel * 2, 3, padding=1),
            Conv2d(channel * 2, channel, 3, padding=1, act=None),
        )
        output_channels = channel
        input_channels = channel + self.corrks**2
        layers = nn.ModuleList()
        layer_dilation = 1
        for _ in range(depth - 1):
            output_channels //= 2
            layers.append(
                Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=layer_dilation,
                    dilation=layer_dilation,
                    norm=norm,
                )
            )
            input_channels = output_channels
            layer_dilation *= 2
        layers.append(
            Conv2d(
                output_channels,
                2,
                kernel_size=3,
                padding=1,
                dilation=1,
                act=None,
                norm=None,
            )
        )
        self.layers = layers
        self.register_buffer(
            "scale",
            torch.tensor([256.0, 256.0]).reshape(1, 2, 1, 1) - 1.0,
            persistent=False,
        )
        self._scale_shape: tuple[int, int] | None = None

    def _local_correlation(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        compressed = self.featcompressor(torch.cat([feature1, feature2], dim=1))
        batch, channels, height, width = feature2.shape
        smooth = KF.gaussian_blur2d(feature1, (13, 13), (3, 3), border_type="constant")
        blocks = F.unfold(
            smooth,
            kernel_size=self.corrks,
            dilation=4,
            padding=2 * (self.corrks - 1),
            stride=1,
        ).reshape(batch, channels, -1, height, width)
        distance = (feature2.unsqueeze(2) - blocks).pow(2).mean(dim=1)
        return torch.cat([compressed, distance], dim=1)

    def forward(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feature1.shape
        features = self.preprocessor(torch.cat([feature1, feature2], dim=0))
        feature1 = features[:batch]
        feature2 = features[batch:]
        if self._scale_shape != (height, width):
            self.scale = torch.tensor(
                [float(width - 1), float(height - 1)],
                device=feature1.device,
                dtype=feature1.dtype,
            ).reshape(1, 2, 1, 1)
            self._scale_shape = (height, width)
        correlation = self._local_correlation(feature1, feature2)
        for layer in self.layers:
            correlation = layer(correlation)
        displacement = KF.gaussian_blur2d(
            correlation, (13, 13), (3, 3), border_type="replicate"
        ).clamp(min=-300, max=300)
        return displacement / self.scale


class DisplacementRefiner(nn.Module):
    def __init__(self, channel: int, dilation: int = 1, depth: int = 4) -> None:
        super().__init__()
        self.preprocessor = nn.Sequential(
            Conv2d(
                channel,
                channel,
                3,
                dilation=dilation,
                padding=dilation,
                norm=None,
                act=None,
            )
        )
        self.featcompressor = nn.Sequential(
            Conv2d(channel * 2, channel * 2, 3, padding=1),
            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=None),
        )
        output_channels = channel
        input_channels = channel + 2
        layer_dilation = 1
        estimator: list[nn.Module] = []
        for _ in range(depth - 1):
            output_channels //= 2
            estimator.append(
                Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=layer_dilation,
                    dilation=layer_dilation,
                    norm=nn.BatchNorm2d,
                )
            )
            input_channels = output_channels
            layer_dilation *= 2
        estimator.append(
            Conv2d(
                output_channels,
                2,
                kernel_size=3,
                padding=1,
                dilation=1,
                act=None,
                norm=None,
            )
        )
        self.estimator = nn.Sequential(*estimator)

    def forward(
        self, feature1: torch.Tensor, feature2: torch.Tensor, displacement: torch.Tensor
    ) -> torch.Tensor:
        batch = feature1.shape[0]
        features = self.preprocessor(torch.cat([feature1, feature2], dim=0))
        features = self.featcompressor(torch.cat([features[:batch], features[batch:]], dim=1))
        return displacement + self.estimator(torch.cat([features, displacement], dim=1))


class DenseMatcher(nn.Module):
    """The SuperFusion image-conditioned dense displacement estimator."""

    def __init__(self, unshare_depth: int = 4, matcher_depth: int = 4) -> None:
        super().__init__()
        self.feature_extractor_unshare1 = FeatureExtractorUnshared(
            unshare_depth, 3, 8, 1, nn.InstanceNorm2d
        )
        self.feature_extractor_unshare2 = FeatureExtractorUnshared(
            unshare_depth, 3, 8, 1, nn.InstanceNorm2d
        )
        base_oc = self.feature_extractor_unshare1.oc
        self.feature_extractor_share1 = nn.Sequential(
            Conv2d(base_oc, base_oc * 2, 3, padding=1, norm=nn.InstanceNorm2d),
            Conv2d(
                base_oc * 2,
                base_oc * 2,
                3,
                stride=2,
                padding=1,
                norm=nn.InstanceNorm2d,
            ),
        )
        self.feature_extractor_share2 = nn.Sequential(
            Conv2d(
                base_oc * 2,
                base_oc * 4,
                3,
                padding=2,
                dilation=2,
                norm=nn.InstanceNorm2d,
            ),
            Conv2d(
                base_oc * 4,
                base_oc * 4,
                3,
                stride=2,
                padding=2,
                dilation=2,
                norm=nn.InstanceNorm2d,
            ),
        )
        self.feature_extractor_share3 = nn.Sequential(
            Conv2d(
                base_oc * 4,
                base_oc * 8,
                3,
                padding=4,
                dilation=4,
                norm=nn.InstanceNorm2d,
            ),
            Conv2d(
                base_oc * 8,
                base_oc * 8,
                3,
                stride=2,
                padding=4,
                dilation=4,
                norm=nn.InstanceNorm2d,
            ),
        )
        self.matcher1 = DisplacementEstimator(base_oc * 4, matcher_depth, dilation=4)
        self.matcher2 = DisplacementEstimator(base_oc * 8, matcher_depth, dilation=2)
        self.refiner = DisplacementRefiner(base_oc * 2, 1)
        self.register_buffer("grid_down", _create_meshgrid(64, 64), persistent=False)
        self.register_buffer("grid_full", _create_meshgrid(128, 128), persistent=False)
        self.register_buffer(
            "scale",
            torch.tensor([128.0, 128.0]).reshape(1, 2, 1, 1) - 1.0,
            persistent=False,
        )
        self._scale_shape: tuple[int, int] | None = None

    @staticmethod
    def _grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        return _create_meshgrid(height, width, device=reference.device, dtype=reference.dtype)

    def _match(
        self,
        feature11: torch.Tensor,
        feature12: torch.Tensor,
        feature21: torch.Tensor,
        feature22: torch.Tensor,
        feature31: torch.Tensor,
        feature32: torch.Tensor,
    ) -> torch.Tensor:
        height, width = feature11.shape[2:]
        if self._scale_shape != (height, width):
            self.scale = torch.tensor(
                [float(width - 1), float(height - 1)],
                device=feature11.device,
                dtype=feature11.dtype,
            ).reshape(1, 2, 1, 1)
            self._scale_shape = (height, width)

        displacement2 = self.matcher2(feature31, feature32)
        displacement2 = F.interpolate(
            displacement2,
            feature21.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        grid_down = self._grid(*feature21.shape[2:], feature21)
        feature21 = F.grid_sample(
            feature21,
            grid_down + displacement2.permute(0, 2, 3, 1),
            align_corners=False,
        )
        displacement1 = self.matcher1(feature21, feature22)
        displacement1 = F.interpolate(
            displacement1,
            feature11.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        displacement2 = F.interpolate(
            displacement2,
            feature11.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        grid_full = self._grid(*feature11.shape[2:], feature11)
        feature11 = F.grid_sample(
            feature11,
            grid_full + (displacement1 + displacement2).permute(0, 2, 3, 1),
            align_corners=False,
        )
        pixels = (displacement1 + displacement2) * self.scale
        displacement = self.refiner(feature11, feature12, pixels)
        return (
            KF.gaussian_blur2d(displacement, (17, 17), (5, 5), border_type="replicate") / self.scale
        )

    def forward(
        self,
        infrared: torch.Tensor,
        visible: torch.Tensor,
        *,
        direction: str = "infrared_to_visible",
    ) -> torch.Tensor:
        """Return a dense sampling field while preserving modality-specific encoders.

        The official checkpoint's first unshared encoder is trained for infrared
        and its second for visible imagery. ``visible_to_infrared`` reverses the
        matching direction after feature extraction; it must not swap the raw
        tensors between those encoders.
        """
        if infrared.shape != visible.shape or infrared.ndim != 4 or infrared.shape[1] != 3:
            raise ValueError("infrared and visible must have equal NCHW shape with three channels")
        if direction not in {"infrared_to_visible", "visible_to_infrared"}:
            raise ValueError(f"unsupported registration direction: {direction}")
        batch = infrared.shape[0]
        feature01 = self.feature_extractor_unshare1(infrared)
        feature02 = self.feature_extractor_unshare2(visible)
        feature1 = self.feature_extractor_share1(torch.cat([feature01, feature02], dim=0))
        feature2 = self.feature_extractor_share2(feature1)
        feature3 = self.feature_extractor_share3(feature2)
        infrared1, visible1 = feature1[:batch], feature1[batch:]
        infrared2, visible2 = feature2[:batch], feature2[batch:]
        infrared3, visible3 = feature3[:batch], feature3[batch:]
        if direction == "infrared_to_visible":
            features = (infrared1, visible1, infrared2, visible2, infrared3, visible3)
            output_size = visible.shape[2:]
        else:
            features = (visible1, infrared1, visible2, infrared2, visible3, infrared3)
            output_size = infrared.shape[2:]
        displacement = self._match(
            *features,
        )
        return F.interpolate(
            displacement,
            output_size,
            mode="bilinear",
            align_corners=False,
        )


def load_superfusion_matcher(checkpoint: str | Path, device: torch.device) -> DenseMatcher:
    """Load the dense matcher from an official SuperFusion checkpoint."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("DM"), dict):
        raise ValueError("SuperFusion checkpoint has no DM state dictionary")
    model = DenseMatcher()
    model.load_state_dict(payload["DM"], strict=True)
    model.eval()
    return model.to(device)


def warp_source_to_target(
    source: torch.Tensor,
    displacement: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Warp source data with a matcher-produced target sampling field."""
    if displacement.ndim != 4 or displacement.shape[1] != 2:
        raise ValueError("displacement must have shape N,2,H,W")
    grid = _create_meshgrid(
        displacement.shape[2],
        displacement.shape[3],
        device=displacement.device,
        dtype=displacement.dtype,
    )
    return F.grid_sample(
        source,
        grid + displacement.permute(0, 2, 3, 1),
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )


def transform_boxes_source_to_target(
    boxes: torch.Tensor,
    displacement: torch.Tensor,
    *,
    perimeter_samples: int = 16,
    inverse_iterations: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform normalized ``cx,cy,w,h`` boxes through a target-to-source field.

    ``grid_sample`` displacement fields are inverse maps: each target location
    identifies a source location. We solve ``target + displacement(target) =
    source`` at sub-pixel perimeter points instead of rasterising small UAV
    boxes into a low-resolution mask. The returned residual is the largest
    normalized-grid inverse error for each image.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape N,4")
    if displacement.ndim != 4 or displacement.shape[:2] != (boxes.shape[0], 2):
        raise ValueError("displacement batch must match N boxes and have two channels")
    if perimeter_samples < 2 or inverse_iterations < 1:
        raise ValueError("perimeter_samples must be >=2 and inverse_iterations must be >=1")

    source = box_perimeter_points(boxes, perimeter_samples)
    target = source.clone()

    for _ in range(inverse_iterations):
        sampled = (
            F.grid_sample(
                displacement,
                target.unsqueeze(2),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(-1)
            .permute(0, 2, 1)
        )
        target = source - sampled

    sampled = (
        F.grid_sample(
            displacement,
            target.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        .squeeze(-1)
        .permute(0, 2, 1)
    )
    residual = torch.linalg.vector_norm(target + sampled - source, dim=-1).amax(dim=1)
    target = (target + 1.0) / 2.0
    minimum = target.amin(dim=1)
    maximum = target.amax(dim=1)
    transformed = torch.cat(((minimum + maximum) / 2.0, maximum - minimum), dim=1)
    return transformed, residual


def box_perimeter_points(boxes: torch.Tensor, perimeter_samples: int = 16) -> torch.Tensor:
    """Return corresponding box-perimeter points in grid coordinates [-1, 1]."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape N,4")
    if perimeter_samples < 2:
        raise ValueError("perimeter_samples must be >=2")
    center = boxes[:, None, :2]
    half_size = boxes[:, None, 2:] / 2.0
    lower = center - half_size
    upper = center + half_size
    steps = torch.linspace(
        0.0,
        1.0,
        perimeter_samples,
        device=boxes.device,
        dtype=boxes.dtype,
    )[None, :, None]
    horizontal = lower + steps * (upper - lower)
    vertical = lower + steps * (upper - lower)
    top = torch.stack((horizontal[..., 0], lower[..., 1].expand_as(horizontal[..., 0])), -1)
    bottom = torch.stack((horizontal[..., 0], upper[..., 1].expand_as(horizontal[..., 0])), -1)
    left = torch.stack((lower[..., 0].expand_as(vertical[..., 1]), vertical[..., 1]), -1)
    right = torch.stack((upper[..., 0].expand_as(vertical[..., 1]), vertical[..., 1]), -1)
    return torch.cat((top, bottom, left, right), dim=1) * 2.0 - 1.0
