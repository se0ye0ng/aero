"""Annotation-free overlapping tiles with explicit pixel-centre coordinates."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int
    input_width: int
    input_height: int

    def restore(self, points):
        scale = np.array([self.input_width / self.width, self.input_height / self.height])
        return (np.asarray(points, dtype=float) + 0.5) / scale - 0.5 + [self.x, self.y]


def prepare_tiles(gray, long_side=640):
    """Nine half-frame tiles, 50% overlap, resized independently to a fixed cap.

    Unlike whole-image preparation, tiles are deliberately upsampled. This tests
    scale/context sensitivity, not the creation of new native image detail.
    """
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 16:
        raise ValueError("uint8 grayscale image with dimensions >=16 required")
    if type(long_side) is not int or long_side < 8 or long_side % 8:
        raise ValueError("long_side must be a positive multiple of eight")
    height, width = gray.shape
    tw, th = (width + 1) // 2, (height + 1) // 2
    xs, ys = [0, (width - tw) // 2, width - tw], [0, (height - th) // 2, height - th]
    factor = long_side / max(tw, th)
    iw, ih = max(8, int(tw * factor) // 8 * 8), max(8, int(th * factor) // 8 * 8)
    result = []
    for y in ys:
        for x in xs:
            tile = Tile(x, y, tw, th, iw, ih)
            image = cv2.resize(
                gray[y : y + th, x : x + tw], (iw, ih), interpolation=cv2.INTER_LINEAR
            )
            result.append((tile, image))
    return result


def merge_exact_matches(points0, points1, confidence):
    """Remove only identical endpoint pairs; highest confidence wins stably.

    Nearby endpoints are not silently merged into a new synthetic correspondence.
    Geometric filtering and qualification remain separate responsibilities.
    """
    p, q, c = (np.asarray(a) for a in (points0, points1, confidence))
    if (
        p.ndim != 2
        or p.shape[1:] != (2,)
        or p.shape != q.shape
        or c.shape != (len(p),)
        or not all(np.isfinite(a).all() for a in (p, q, c))
    ):
        raise ValueError("finite endpoint pairs and confidences required")
    order = np.argsort(-c, kind="stable")
    _, first = np.unique(np.c_[p[order], q[order]], axis=0, return_index=True)
    keep = order[np.sort(first)]
    return p[keep], q[keep], c[keep], keep


def match_tiles(infer_pair, gray0, gray1, progress=None):
    """Evaluate all 9x9 tile pairs; no target annotations or diagonal-only shortcut."""
    first, second = prepare_tiles(gray0), prepare_tiles(gray1)
    points0, points1, scores, records = [], [], [], []
    for i, (a, image0) in enumerate(first):
        for j, (b, image1) in enumerate(second):
            p, q, c, flags = infer_pair(image0, image1)
            p, q = a.restore(p), b.restore(q)
            valid = np.isfinite(p).all(1) & np.isfinite(q).all(1) & np.isfinite(c)
            for xy, image in ((p, gray0), (q, gray1)):
                valid &= (xy >= 0).all(1) & (xy <= np.array(image.shape[::-1]) - 1).all(1)
            points0.append(p[valid])
            points1.append(q[valid])
            scores.append(c[valid])
            records.append(dict(tile0=i, tile1=j, retained=int(valid.sum()), flags=flags))
        if progress is not None:
            progress((i + 1) * len(second), len(first) * len(second))
    p, q, c, _ = merge_exact_matches(
        np.concatenate(points0), np.concatenate(points1), np.concatenate(scores)
    )
    return (
        p,
        q,
        c,
        dict(
            tile_pairs=len(records),
            records=records,
            before_exact_dedup=sum(len(x) for x in points0),
        ),
    )
