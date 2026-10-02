import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from aero_ir.registration.silhouette import silhouette_scores
from scripts.probe_registration_silhouette import run


def inputs():
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[17:28, 18:27] = 1
    y, x = np.meshgrid(np.arange(48) + 0.5, np.arange(48) + 0.5, indexing="ij")
    grid = np.stack((2 * x / 48 - 1, 2 * y / 48 - 1), axis=-1).astype(np.float32)
    maps = np.stack([grid, grid + [2 * 4 / 48, 0], grid - [2 * 4 / 48, 0]])
    return mask, maps


def test_known_translation():
    source, maps = inputs()
    target = np.roll(source, -4, axis=1)
    result = silhouette_scores(source, target, maps)
    assert all(r["eligible"] for r in result["candidates"])
    for metric in ("dice_loss", "symmetric_boundary_chamfer_px"):
        values = [r[metric] for r in result["candidates"]]
        assert np.argmin(values) == 1
        assert abs(values[1]) < 1e-5
    assert result["independent_pixel_correspondence_accuracy"] is False


def test_empty_masks_abstain():
    mask, maps = inputs()
    result = silhouette_scores(mask * 0, mask * 0, maps)
    assert all(not r["eligible"] and r["dice_loss"] is None for r in result["candidates"])


@pytest.mark.parametrize("kind", ("nan_mask", "nan_map", "shape", "soft_mask"))
def test_invalid_inputs(kind):
    mask, maps = inputs()
    mask = mask.astype(float)
    if kind == "nan_mask":
        mask[0, 0] = np.nan
    elif kind == "nan_map":
        maps[0, 0, 0, 0] = np.nan
    elif kind == "shape":
        maps = maps[:, :-1]
    else:
        mask[0, 0] = 0.5
    with pytest.raises(ValueError):
        silhouette_scores(mask, mask, maps)


def test_clipped_contours_abstain():
    mask, maps = inputs()
    mask[:20, :20] = 1
    result = silhouette_scores(mask, mask, maps)
    assert all(not r["eligible"] for r in result["candidates"])


def manifest_fixture(tmp_path):
    mask, maps = inputs()
    Image.fromarray(mask * 255).save(tmp_path / "image.png")
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "maps.npy", maps)
    (tmp_path / "checkpoint.pth").write_bytes(b"synthetic test provenance only")

    def item(name):
        return {"path": name, "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}

    side = {
        "frame_id": "synthetic-test-frame",
        "image_grid": "native",
        "image": item("image.png"),
        "mask": item("mask.npy"),
    }
    return {
        "schema_version": 1,
        "pair_id": "synthetic-test-pair",
        "split": "train",
        "validation_or_test_access": "none",
        "review": {
            "approved": True,
            "same_physical_outline": True,
            "reviewer_id": "test-fixture-not-real-review",
            "reviewed_at": "2026-09-20",
        },
        "source": side,
        "target": side,
        "checkpoint": item("checkpoint.pth"),
        "maps": item("maps.npy"),
        "map_convention": "target_to_source_normalized_pixel_centers_align_corners_false",
        "candidate_generation_uses_masks": False,
        "baseline_index": 0,
        "candidate_specs": [{"is_unchanged": True}, {"dx": 4}, {"dx": -4}],
    }


def test_same_mask_cannot_claim_independent_accuracy(tmp_path):
    manifest = manifest_fixture(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    report = run(path)
    assert report["selection"]["dice_loss"] == 0
    assert report["result"]["independent_pixel_correspondence_accuracy"] is False
    assert report["qualification"] == "hold_not_independent_pixel_correspondence"


@pytest.mark.parametrize(
    "kind", ("unapproved", "unreviewed", "different_outline", "wrong_hash", "test_split")
)
def test_manifest_fail_closed(tmp_path, kind):
    manifest = manifest_fixture(tmp_path)
    if kind == "unapproved":
        manifest["review"]["approved"] = False
    elif kind == "unreviewed":
        del manifest["review"]["reviewer_id"]
    elif kind == "different_outline":
        manifest["review"]["same_physical_outline"] = False
    elif kind == "wrong_hash":
        manifest["maps"]["sha256"] = "0" * 64
    else:
        manifest["split"] = "test"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run(path)


@pytest.mark.parametrize("field", ("reviewer_id", "reviewed_at"))
@pytest.mark.parametrize("value", (None, 0, False, [], "", "   "))
def test_review_identity_requires_real_nonempty_string(tmp_path, field, value):
    manifest = manifest_fixture(tmp_path)
    manifest["review"][field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run(path)
