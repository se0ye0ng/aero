"""Per-image and per-object radiometric statistics.

Every function returns a 1-D array of per-object or per-image samples. Distributions, not
scalars: the comparison between a generated set and a real set is distributional, because a
generator can match a mean perfectly while destroying the spread that a detector relies on.

Images are expected as float arrays in a consistent intensity unit (DN or normalised), shape
(H, W). Boxes are ``(x, y, w, h)`` in pixels, COCO convention.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

COMPONENTS = {
    "R1": "target_background_delta",
    "R2": "thermal_polarity",
    "R3": "target_snr",
    "R4": "radial_power_spectrum",
    "R5": "noise_psd",
    "R6": "target_pixel_area",
    "R7": "dynamic_range",
    "R8": "column_structure_energy",
}


def _box_slices(box, shape):
    x, y, w, h = (int(round(v)) for v in box)
    y0, y1 = max(0, y), min(shape[0], y + h)
    x0, x1 = max(0, x), min(shape[1], x + w)
    return (slice(y0, y1), slice(x0, x1))


def _annulus_mask(shape, box, dilation: int) -> np.ndarray:
    """Background annulus: a dilated box minus the box itself."""
    inner = np.zeros(shape, dtype=bool)
    inner[_box_slices(box, shape)] = True
    outer = ndimage.binary_dilation(inner, iterations=max(1, dilation))
    return outer & ~inner


def target_background_delta(image, boxes, annulus_dilation_px: int = 8) -> np.ndarray:
    """R1. Median intensity inside each box minus the median of its background annulus.

    The primary detection signal. A generator that compresses this makes targets
    undetectable no matter how realistic the image looks to a human.
    """
    out = []
    for box in boxes:
        sl = _box_slices(box, image.shape)
        tgt = image[sl]
        if tgt.size == 0:
            continue
        bg = image[_annulus_mask(image.shape, box, annulus_dilation_px)]
        if bg.size == 0:
            continue
        out.append(float(np.median(tgt) - np.median(bg)))
    return np.asarray(out, dtype=np.float64)


def thermal_polarity(image, boxes, annulus_dilation_px: int = 8) -> np.ndarray:
    """R2. Sign of R1 per object: +1 hot-on-cold, -1 cold-on-hot, 0 indeterminate."""
    d = target_background_delta(image, boxes, annulus_dilation_px)
    return np.sign(d)


def target_snr(image, boxes, annulus_dilation_px: int = 8) -> np.ndarray:
    """R3. R1 divided by the background standard deviation in the annulus.

    Separates low contrast from high noise. Two very different failure modes that a single
    contrast number conflates.
    """
    out = []
    for box in boxes:
        sl = _box_slices(box, image.shape)
        tgt = image[sl]
        if tgt.size == 0:
            continue
        bg = image[_annulus_mask(image.shape, box, annulus_dilation_px)]
        if bg.size < 8:
            continue
        sigma = float(np.std(bg))
        if sigma <= 1e-9:
            continue
        out.append(float(np.median(tgt) - np.median(bg)) / sigma)
    return np.asarray(out, dtype=np.float64)


def radial_power_spectrum(image, n_bins: int = 64) -> np.ndarray:
    """R4. Azimuthally averaged log power spectral density.

    Captures optical blur and, more importantly, over-sharpening: generators routinely
    synthesise edge energy at spatial frequencies no infrared optic can pass.
    """
    img = np.asarray(image, dtype=np.float64)
    img = img - img.mean()
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    f = np.fft.fftshift(np.fft.fft2(img * win))
    power = np.abs(f) ** 2
    cy, cx = (s // 2 for s in power.shape)
    yy, xx = np.indices(power.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(min(cy, cx))
    bins = np.linspace(0, r_max, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    valid = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[valid], weights=power.ravel()[valid], minlength=n_bins)
    counts = np.bincount(idx[valid], minlength=n_bins).clip(min=1)
    return np.log10((sums / counts) + 1e-12)


def noise_psd(image, highpass_sigma_px: float = 1.5, n_bins: int = 32) -> np.ndarray:
    """R5. Radial PSD of the high-pass residual.

    Generated imagery is typically too clean. A detector trained on it has never had to
    separate a target from a sensor noise floor, and fails when it meets one.
    """
    img = np.asarray(image, dtype=np.float64)
    residual = img - ndimage.gaussian_filter(img, sigma=highpass_sigma_px)
    return radial_power_spectrum(residual, n_bins=n_bins)


def target_pixel_area(boxes) -> np.ndarray:
    """R6. Box area in pixels. The small-target tail is where operational detection lives."""
    return np.asarray([float(b[2]) * float(b[3]) for b in boxes], dtype=np.float64)


def dynamic_range(image) -> np.ndarray:
    """R7. Percentile spread, entropy and clipped fraction of the intensity histogram."""
    img = np.asarray(image, dtype=np.float64).ravel()
    p1, p99 = np.percentile(img, [1.0, 99.0])
    hist, _ = np.histogram(img, bins=256, density=True)
    p = hist / max(hist.sum(), 1e-12)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    lo, hi = img.min(), img.max()
    clipped = float(((img <= lo + 1e-9) | (img >= hi - 1e-9)).mean())
    return np.asarray([float(p99 - p1), entropy, clipped], dtype=np.float64)


def column_structure_energy(image, highpass_sigma_px: float = 1.5) -> np.ndarray:
    """R8. Column-wise over row-wise variance of the high-pass residual.

    Real focal-plane arrays leave residual column structure after non-uniformity correction.
    Generated images do not, and the ratio is a clean one-number tell.
    """
    img = np.asarray(image, dtype=np.float64)
    residual = img - ndimage.gaussian_filter(img, sigma=highpass_sigma_px)
    col_var = float(np.var(residual.mean(axis=0)))
    row_var = float(np.var(residual.mean(axis=1)))
    return np.asarray([col_var / max(row_var, 1e-12)], dtype=np.float64)


def image_set_statistics(images, boxes_per_image, cfg) -> dict[str, np.ndarray]:
    """Accumulate every enabled component over a set of images.

    Args:
        images: iterable of (H, W) float arrays.
        boxes_per_image: iterable of box lists aligned with ``images``.
        cfg: object exposing ``components``, ``annulus_dilation_px``, ``highpass_sigma_px``.

    Returns:
        Mapping from component id to the pooled sample array.
    """
    wanted = set(getattr(cfg, "components", COMPONENTS.keys()))
    ann = int(getattr(cfg, "annulus_dilation_px", 8))
    hp = float(getattr(cfg, "highpass_sigma_px", 1.5))
    acc: dict[str, list[np.ndarray]] = {k: [] for k in wanted}

    for img, boxes in zip(images, boxes_per_image, strict=False):
        img = np.asarray(img, dtype=np.float64)
        if "R1" in wanted:
            acc["R1"].append(target_background_delta(img, boxes, ann))
        if "R2" in wanted:
            acc["R2"].append(thermal_polarity(img, boxes, ann))
        if "R3" in wanted:
            acc["R3"].append(target_snr(img, boxes, ann))
        if "R4" in wanted:
            acc["R4"].append(radial_power_spectrum(img))
        if "R5" in wanted:
            acc["R5"].append(noise_psd(img, hp))
        if "R6" in wanted:
            acc["R6"].append(target_pixel_area(boxes))
        if "R7" in wanted:
            acc["R7"].append(dynamic_range(img))
        if "R8" in wanted:
            acc["R8"].append(column_structure_energy(img, hp))

    return {
        k: (np.concatenate(v) if v else np.asarray([], dtype=np.float64))
        for k, v in acc.items()
    }
