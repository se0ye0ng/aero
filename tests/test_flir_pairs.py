import copy
import json

import numpy as np

from aero_ir.data.flir_pairs import (
    audit_video_pair_registration,
    build_video_pair_manifest,
    verify_registration_audit,
    verify_video_pair_manifest,
)
from aero_ir.utils.manifest import canonical_hash


def _write_video_pair_release(root, *, unmatched_second_sequence=False):
    rgb_images = []
    thermal_images = []
    rgb_annotations = []
    thermal_annotations = []
    mapping = {}
    annotation_id = 1

    for sequence_index in range(2):
        rgb_video = f"rgb{sequence_index}"
        thermal_video = f"thermal{sequence_index}"
        for frame_index in range(2):
            token = f"token{sequence_index}{frame_index}"
            rgb_name = f"video-{rgb_video}-frame-{frame_index:06d}-{token}.jpg"
            thermal_name = f"video-{thermal_video}-frame-{frame_index:06d}-{token}.jpg"
            rgb_id = 100 * sequence_index + frame_index + 1
            thermal_id = 1000 + rgb_id
            rgb_images.append(
                {"id": rgb_id, "file_name": f"data/{rgb_name}", "width": 100, "height": 80}
            )
            thermal_images.append(
                {
                    "id": thermal_id,
                    "file_name": f"data/{thermal_name}",
                    "width": 50,
                    "height": 40,
                }
            )
            rgb_track = sequence_index + 1
            thermal_track = rgb_track
            if unmatched_second_sequence and sequence_index == 1:
                thermal_track += 100
            rgb_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": rgb_id,
                    "category_id": 1,
                    "track_id": rgb_track,
                    "bbox": [20, 16, 40, 32],
                }
            )
            annotation_id += 1
            thermal_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": thermal_id,
                    "category_id": 1,
                    "track_id": thermal_track,
                    "bbox": [10, 8, 20, 16],
                }
            )
            annotation_id += 1
            mapping[rgb_name] = thermal_name

            rgb_path = root / "video_rgb_test" / "data" / rgb_name
            thermal_path = root / "video_thermal_test" / "data" / thermal_name
            analytics_path = (
                root
                / "video_thermal_test"
                / "analyticsData"
                / thermal_name.replace(".jpg", ".tiff")
            )
            for path in (rgb_path, thermal_path, analytics_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

    categories = [{"id": 1, "name": "person"}]
    rgb_coco = {
        "images": rgb_images,
        "annotations": rgb_annotations,
        "categories": categories,
    }
    thermal_coco = {
        "images": thermal_images,
        "annotations": thermal_annotations,
        "categories": categories,
    }
    (root / "video_rgb_test" / "coco.json").write_text(json.dumps(rgb_coco), encoding="utf-8")
    (root / "video_thermal_test" / "coco.json").write_text(
        json.dumps(thermal_coco), encoding="utf-8"
    )
    map_path = root.parent / "rgb_to_thermal_vid_map.json"
    map_path.write_text(json.dumps(mapping), encoding="utf-8")
    return map_path


def _release_audit(root):
    return {
        "root": str(root.resolve()),
        "source_archive": {"name": "fixture.zip", "sha256": "fixture"},
        "gates": {"time_synchronised_video_pairs": "pass"},
    }


def test_video_pair_manifest_freezes_complete_official_map(tmp_path):
    root = tmp_path / "FLIR_ADAS_v2"
    map_path = _write_video_pair_release(root)

    registration = audit_video_pair_registration(root, map_path)
    manifest = build_video_pair_manifest(
        root,
        map_path,
        registration_audit=registration,
        release_audit=_release_audit(root),
    )

    assert registration["gates"] == {
        "registration_evidence": "pass",
        "direct_label_transfer": "pass",
        "training_use": "hold",
    }
    assert manifest["counts"] == {
        "pairs": 4,
        "sequence_pairs": 2,
        "unique_rgb_images": 4,
        "unique_thermal_images": 4,
    }
    assert "detector or generator training" in manifest["prohibited_uses"]
    verify_registration_audit(registration)
    verify_video_pair_manifest(manifest)


def test_registration_audit_holds_when_sequence_coverage_is_incomplete(tmp_path):
    root = tmp_path / "FLIR_ADAS_v2"
    map_path = _write_video_pair_release(root, unmatched_second_sequence=True)

    registration = audit_video_pair_registration(root, map_path)

    assert registration["counts"]["sequence_pairs"] == 2
    assert registration["counts"]["sequences_with_shared_track_category_keys"] == 1
    assert registration["gates"]["registration_evidence"] == "pass"
    assert registration["gates"]["direct_label_transfer"] == "hold"
    assert registration["gates"]["training_use"] == "hold"


def test_pair_manifest_and_registration_hashes_reject_tampering(tmp_path):
    root = tmp_path / "FLIR_ADAS_v2"
    map_path = _write_video_pair_release(root)
    registration = audit_video_pair_registration(root, map_path)
    manifest = build_video_pair_manifest(
        root,
        map_path,
        registration_audit=registration,
        release_audit=_release_audit(root),
    )

    tampered_registration = copy.deepcopy(registration)
    tampered_registration["gates"]["training_use"] = "pass"
    with np.testing.assert_raises_regex(ValueError, "hash mismatch"):
        verify_registration_audit(tampered_registration)

    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["records"][0]["frame_index"] = 999
    with np.testing.assert_raises_regex(ValueError, "hash mismatch"):
        verify_video_pair_manifest(tampered_manifest)

    rehashed_manifest = copy.deepcopy(manifest)
    rehashed_manifest["prohibited_uses"].remove("detector or generator training")
    unsigned = {
        key: value for key, value in rehashed_manifest.items() if key != "pair_manifest_sha256"
    }
    rehashed_manifest["pair_manifest_sha256"] = canonical_hash(unsigned)
    with np.testing.assert_raises_regex(ValueError, "prohibitions are incomplete"):
        verify_video_pair_manifest(rehashed_manifest)
