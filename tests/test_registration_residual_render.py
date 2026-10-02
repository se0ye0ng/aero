import json
from types import SimpleNamespace

import numpy as np
import pytest

from aero_ir.utils.manifest import file_sha256
from scripts import render_registration_residual as render


@pytest.mark.parametrize("defect", [None, "pixels", "box", "annotation", "position"])
def test_native_decode_must_reproduce_cached_observation(tmp_path, monkeypatch, defect):
    root, cache, sid = tmp_path / "data", tmp_path / "cache", "sequence"
    folder = root / "train" / sid
    folder.mkdir(parents=True)
    shard = {"pairs": 3}
    for name in ("visible", "infrared"):
        path = folder / f"{name}.json"
        path.write_text("{}")
        shard[f"{name}_annotations_sha256"] = file_sha256(path)
    shard_path = cache / "shards" / sid / "manifest.json"
    shard_path.parent.mkdir(parents=True)
    shard_path.write_text(json.dumps(shard))
    rgb = np.full((32, 32, 3), [30, 70, 90], np.uint8)
    box = np.asarray([0.5, 0.5, 0.2, 0.2], np.float32)
    arrays = {
        "visible": rgb[None].copy(),
        "infrared": rgb[None].copy(),
        "source_boxes": box[None].copy(),
        "target_boxes": box[None].copy(),
    }
    sample = {"selection": [[sid, 1]]}
    monkeypatch.setattr(
        render, "_candidate_pairs", lambda folder: ([2, 4, 6], [box] * 3, [box] * 3)
    )
    monkeypatch.setattr(
        render.cv2, "VideoCapture", lambda path: SimpleNamespace(release=lambda: None)
    )

    def read(capture, frame, path):
        assert frame == 4
        return rgb[..., ::-1].copy()

    monkeypatch.setattr(render, "_read_at", read)
    if defect == "pixels":
        arrays["visible"][0, 0, 0, 0] += 1
    elif defect == "box":
        arrays["source_boxes"][0, 0] += 0.01
    elif defect == "annotation":
        (folder / "visible.json").write_text('{"changed": true}')
    elif defect == "position":
        sample["selection"][0][1] = 0
    if defect:
        with pytest.raises(ValueError):
            render.native_images(root, cache, sid, sample, arrays)
    else:
        images, metadata = render.native_images(root, cache, sid, sample, arrays)
        assert all(np.array_equal(image, rgb) for image in images)
        assert metadata["native_frame_index"] == 4
        assert len(metadata["metadata"]["visible"]["decoded_native_rgb_sha256"]) == 64
