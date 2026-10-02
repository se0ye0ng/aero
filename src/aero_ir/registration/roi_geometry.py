"""Isotropic, box-centred pixel-centre crops with explicit missing-data support."""

import cv2
import numpy as np


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError("expected Nx2 points and 3x3 matrix")
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError("nonfinite geometry")
    values = np.c_[points, np.ones(len(points))] @ matrix.T
    if (np.abs(values[:, 2]) < 1e-10).any():
        raise ValueError("projection at infinity")
    return values[:, :2] / values[:, 2:]


def centred_transform(box, side=256):
    """Do not clip the requested crop; missing camera pixels become invalid padding.

    Native box centre maps exactly to (side-1)/2. Both axes have the same scale.
    This is a supervised ROI diagnostic: existing train boxes define the crop.
    """
    box = np.asarray(box, dtype=float)
    if (
        box.shape != (4,)
        or not np.isfinite(box).all()
        or (box[2:] <= box[:2]).any()
        or side < 32
        or side % 8
    ):
        raise ValueError("invalid box or output size")
    centre = (box[:2] + box[2:]) / 2
    extent = max(float((box[2:] - box[:2]).max()) * 3, 48.0)
    scale = side / extent
    matrix = np.eye(3)
    matrix[:2, :2] *= scale
    matrix[:2, 2] = (side - 1) / 2 - scale * centre
    return matrix


def legacy_transform(metadata):
    scale = np.asarray(metadata["scale_xy"], dtype=float)
    origin = np.asarray(metadata["bounds_xyxy"][:2], dtype=float)
    if scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid legacy scale")
    matrix = np.diag([*scale, 1.0])
    matrix[:2, 2] = scale * (0.5 - origin) - 0.5
    return matrix


def warp(array, matrix, side=256, nearest=False, fill=0):
    return cv2.warpAffine(
        np.asarray(array),
        np.asarray(matrix)[:2],
        (side, side),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def prepare_centred(gray, box, hud, side=256):
    if gray.ndim != 2 or gray.dtype != np.uint8 or hud.shape != gray.shape:
        raise ValueError("expected grayscale image and same-size HUD mask")
    matrix = centred_transform(box, side)
    image = warp(gray, matrix, side, fill=float(np.median(gray)))
    valid = warp(np.ones(gray.shape, np.float32), matrix, side) >= 1 - 1e-6
    masked = warp(hud.astype(np.float32), matrix, side) > 0
    # Inpaint HUD only, not missing camera content. Fill values never become evidence.
    clean = cv2.inpaint(image, (masked & valid).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    excluded = cv2.dilate((masked | ~valid).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(
        bool
    )
    cb = transform_points(np.asarray(box).reshape(2, 2), matrix).ravel()
    return {
        "raw": image,
        "inpaint": clean,
        "valid": valid,
        "excluded": excluded,
        "box": cb,
        "meta": {
            "native_to_crop": matrix.tolist(),
            "crop_to_native": np.linalg.inv(matrix).tolist(),
            "native_box": np.asarray(box).tolist(),
            "output_side": side,
            "isotropic": True,
            "valid_fraction": float(valid.mean()),
        },
    }


def restore_cached(masks, image, excluded, prompt_box, metadata):
    """Undo old anisotropic crop, then re-centre. Does NOT rerun SAM or invent data."""
    side = image.shape[0]
    old = legacy_transform(metadata)
    box = transform_points(np.asarray(prompt_box).reshape(2, 2), np.linalg.inv(old)).ravel()
    new = centred_transform(box, side)
    correction = new @ np.linalg.inv(old)
    valid = warp(np.ones(image.shape, np.float32), correction, side) >= 1 - 1e-6
    support = warp((~excluded).astype(np.float32), correction, side) >= 1 - 1e-6
    return {
        "masks": np.stack(
            [warp(m.astype(np.uint8), correction, side, nearest=True).astype(bool) for m in masks]
        ),
        "image": warp(image, correction, side),
        "valid": valid,
        "excluded": ~support,
        "box": transform_points(box.reshape(2, 2), new).ravel(),
        "meta": {
            "native_to_crop": new.tolist(),
            "crop_to_native": np.linalg.inv(new).tolist(),
            "native_box": box.tolist(),
            "output_side": side,
            "cached_crop_to_corrected_crop": correction.tolist(),
            "isotropic": True,
            "sam_rerun": False,
        },
    }
