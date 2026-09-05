import json
from pathlib import Path

from aero_ir.data.antiuav import audit_release


def _write_sequence(root: Path, split: str, sequence: str, frames: int = 3) -> None:
    sequence_root = root / split / sequence
    sequence_root.mkdir(parents=True)
    visible = {"exist": [1] * frames, "gt_rect": [[100, 100, 20, 20]] * frames}
    infrared = {"exist": [1] * frames, "gt_rect": [[30, 40, 10, 12]] * frames}
    (sequence_root / "visible.json").write_text(json.dumps(visible), encoding="utf-8")
    (sequence_root / "infrared.json").write_text(json.dumps(infrared), encoding="utf-8")
    (sequence_root / "visible.mp4").touch()
    (sequence_root / "infrared.mp4").touch()


def _fake_probe(path: Path) -> dict:
    visible = path.name == "visible.mp4"
    return {
        "opened": True,
        "frames": 3,
        "width": 1920 if visible else 640,
        "height": 1080 if visible else 512,
        "fps": 20.0,
    }


def test_release_audit_separates_integrity_and_registration_gates(tmp_path):
    root = tmp_path / "Anti-UAV300"
    labels = root / "label_new"
    labels.mkdir(parents=True)
    for split, sequence in (("train", "train-seq"), ("val", "val-seq"), ("test", "test-seq")):
        (labels / f"{split}.json").write_text(json.dumps({sequence: ["SV"]}), encoding="utf-8")
        _write_sequence(root, split, sequence)

    report = audit_release(root, video_probe=_fake_probe)

    assert report["gates"]["extraction_completeness"] == "pass"
    assert report["gates"]["annotation_integrity"] == "pass"
    assert report["gates"]["official_manifest_sequence_disjointness"] == "pass"
    assert report["gates"]["paired_temporal_integrity"] == "pass"
    assert report["gates"]["train_only_pilot_eligible"] == "pass"
    assert report["gates"]["direct_rgb_to_ir_label_transfer"] == "hold"
    assert report["gates"]["archive_integrity"] == "hold"


def test_missing_validation_extraction_holds_full_gate_but_not_train_pilot(tmp_path):
    root = tmp_path / "Anti-UAV300"
    labels = root / "label_new"
    labels.mkdir(parents=True)
    for split, sequence in (("train", "train-seq"), ("val", "val-seq"), ("test", "test-seq")):
        (labels / f"{split}.json").write_text(json.dumps({sequence: ["SV"]}), encoding="utf-8")
        if split != "val":
            _write_sequence(root, split, sequence)

    report = audit_release(root, video_probe=_fake_probe)

    assert report["gates"]["extraction_completeness"] == "hold"
    assert report["gates"]["paired_temporal_integrity"] == "hold"
    assert report["gates"]["train_only_pilot_eligible"] == "pass"
    assert report["splits"]["val"]["missing_sequences"] == ["val-seq"]
