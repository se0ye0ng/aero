"""Deterministic simulated-IR baseline - the third training arm.

The comparison this repository has to support is three-way, not two-way:

    real                    the floor
    real + simulated        what a physics/CG simulator already gives you, for free
    real + generated        what a generative model gives you on top

Without the middle arm there is no way to say the generative step *earned* anything: a gain
over real-only could be a gain any crude simulation would also produce, and a loss could be a
loss any non-real imagery would produce. The generative model is only interesting to the
extent it beats this baseline.

This class is a minimal simulator in the sense a rendering pipeline is: it knows object
identity from the labels and assigns radiometry accordingly, rather than learning a mapping.

    1. background apparent temperature from a smoothed luminance proxy
    2. class-conditional temperature offsets applied inside labelled regions
    3. the same :class:`~aero_ir.sensor.pipeline.IRSensorPipeline` every other arm goes through

It is deliberately crude. That is the point: it is the "free" option any programme already
has, and it is public, deterministic and reproducible.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Apparent temperature offsets in kelvin above local background. Coarse, public, and
# documented rather than tuned - a tuned baseline would not be a baseline.
DEFAULT_CLASS_DELTA_K = {
    "person": 6.0,
    "bike": 2.0,
    "car": 4.0,
    "motor": 4.5,
    "bus": 3.5,
    "truck": 3.5,
    "uav": 2.5,
    "_default": 3.0,
}


class SyntheticBaselineGenerator:
    """Render a pseudo-IR image from a visible image plus its labels."""

    name = "synthetic_baseline"

    def __init__(
        self,
        class_delta_k: dict[str, float] | None = None,
        background_smooth_px: float = 6.0,
        base_temp_k: float = 290.0,
        scene_contrast_k: float = 8.0,
        scene_response_dn_per_K: float = 12.0,
        edge_softness_px: float = 1.5,
        sensor=None,
        seed: int = 0,
    ) -> None:
        self.class_delta_k = dict(class_delta_k or DEFAULT_CLASS_DELTA_K)
        self.background_smooth_px = float(background_smooth_px)
        self.base_temp_k = float(base_temp_k)
        self.scene_contrast_k = float(scene_contrast_k)
        self.scene_response_dn_per_K = float(scene_response_dn_per_K)
        self.edge_softness_px = float(edge_softness_px)
        self.sensor = sensor
        self.rng = np.random.default_rng(seed)

    def _luminance(self, rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.float64)
        if arr.ndim == 2:
            lum = arr
        else:
            lum = arr[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        lo, hi = np.percentile(lum, [1.0, 99.0])
        return np.clip((lum - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

    def _delta_for(self, label) -> float:
        return self.class_delta_k.get(str(label), self.class_delta_k["_default"])

    def render(self, rgb: np.ndarray, boxes, labels=None) -> np.ndarray:
        """Return a pseudo-IR image in digital numbers."""
        lum = self._luminance(rgb)
        # Luminance is a weak proxy for apparent temperature: it carries scene structure but
        # not thermal identity, which is exactly why the class offsets below are needed.
        temp = self.base_temp_k + self.scene_contrast_k * ndimage.gaussian_filter(
            lum, self.background_smooth_px
        )

        labels = labels if labels is not None else ["_default"] * len(boxes)
        target = np.zeros_like(temp)
        for box, label in zip(boxes, labels, strict=False):
            x, y, bw, bh = (int(round(v)) for v in box)
            y0, y1 = max(0, y), min(temp.shape[0], y + bh)
            x0, x1 = max(0, x), min(temp.shape[1], x + bw)
            if y1 <= y0 or x1 <= x0:
                continue
            target[y0:y1, x0:x1] = self._delta_for(label)

        # Soften box edges: a rendered target does not have a rectangular thermal boundary,
        # and a hard rectangle would hand the detector a trivially learnable artifact.
        target = ndimage.gaussian_filter(target, self.edge_softness_px)
        return (temp + target) * self.scene_response_dn_per_K

    def generate(self, sources, labels, **kwargs):
        images, provenance = [], []
        for i, (rgb, ann) in enumerate(zip(sources, labels, strict=False)):
            boxes = ann["boxes"] if isinstance(ann, dict) else ann
            classes = ann.get("labels") if isinstance(ann, dict) else None
            images.append(self.render(rgb, boxes, classes))
            provenance.append({"source_id": i, "generator": self.name, "learned": False})
        if self.sensor is not None:
            images = self.sensor.apply_numpy(images)
        return images, labels, provenance
