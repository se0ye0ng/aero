"""Prepare a small Korean, model-blind part-familiarization exercise, never GT."""

import argparse
import json
import math
import shutil
import tarfile
from pathlib import Path

from PIL import Image

from aero_ir.utils.manifest import file_sha256
from aero_ir.utils.paths import antiuav300_root

# Six fixed midpoint frames across the previous sixteen sequences, not quality selected.
PANEL_INDICES = (1, 10, 19, 28, 37, 46)


def crop_box(rect, width, height):
    if len(rect) != 4 or not all(math.isfinite(v) for v in rect):
        raise ValueError("invalid dataset box")
    x, y, w, h = rect
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("box outside image or empty")
    # Context helps readers distinguish the body, arms and any nearby HUD.
    mx, my = max(24, w * 0.75), max(24, h * 0.75)
    return [
        max(0, math.floor(x - mx)),
        max(0, math.floor(y - my)),
        min(width, math.ceil(x + w + mx)),
        min(height, math.ceil(y + h + my)),
    ]


def render(template, manifest, digest, slot):
    if slot not in ("A", "B"):
        raise ValueError("invalid reviewer slot")
    return (
        template.replace("__MANIFEST__", json.dumps(manifest).replace("<", "\\u003c"))
        .replace("__MANIFEST_HASH__", digest)
        .replace("__SLOT__", slot)
    )


def prepare(panel, root, output):
    if output.exists():
        raise FileExistsError("choose a new practice directory")
    panel_file = panel / "manifest.json"
    old = json.loads(panel_file.read_text())
    if old.get("schema") != "aero_physical_panel_v1" or old.get("split") != "train":
        raise ValueError("requires original train-only review panel")
    if len(old["pairs"]) != 48:
        raise ValueError("expected immutable48-pair source panel")
    if file_sha256(root / "label_new/train.json") != old["train_manifest_sha256"]:
        raise ValueError("train split changed")
    project = Path(__file__).resolve().parents[1]
    template_path = project / "assets/registration_practice.html"
    guide = project / "docs/registration_practice_ko.md"
    manifest = {
        "schema": "aero_part_familiarization_panel_v1",
        "split": "train",
        "purpose": "practice_only_not_registration_gt",
        "source_panel_sha256": file_sha256(panel_file),
        "selection_indices": PANEL_INDICES,
        "selection": "fixed six of48: midpoint frames, no image-quality selection",
        "box_usage": "official boxes for display zoom only, not correspondence labels",
        "coordinate_convention": "original pixel centres, top-left centre0,0",
        "sources": {
            str(p.relative_to(project)): file_sha256(p)
            for p in (Path(__file__), template_path, guide)
        },
        "pairs": [],
    }
    planned = []
    for i in PANEL_INDICES:
        src = old["pairs"][i]
        seq = src["sequence_id"]
        if Path(seq).name != seq:
            raise ValueError("invalid sequence ID")
        pair = {k: src[k] for k in ("pair_id", "sequence_id", "frame_index")}
        pair["images"] = {}
        for modality in ("visible", "infrared"):
            entry = src["images"][modality]
            image = (panel / entry["path"]).resolve()
            if not image.is_relative_to(panel.resolve()) or file_sha256(image) != entry["sha256"]:
                raise ValueError("source image path/hash mismatch")
            annpath = root / "train" / seq / f"{modality}.json"
            if file_sha256(annpath) != entry["annotation_sha256"]:
                raise ValueError("annotations changed")
            annotation = json.loads(annpath.read_text())
            frame = src["frame_index"]
            if annotation["exist"][frame] != 1:
                raise ValueError("practice pair requires an existing annotated target")
            with Image.open(image) as im:
                width, height = im.size
            if [height, width] != entry["shape"][:2]:
                raise ValueError("image dimensions changed")
            relative = f"images/{image.name}"
            pair["images"][modality] = {
                "path": relative,
                "shape": entry["shape"],
                "sha256": entry["sha256"],
                "annotation_sha256": entry["annotation_sha256"],
                "practice_crop_xyxy": crop_box(annotation["gt_rect"][frame], width, height),
            }
            planned.append((image, relative))
        manifest["pairs"].append(pair)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for source, relative in planned:
        shutil.copyfile(source, output / relative)
    destination = output / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for slot in ("A", "B"):
        (output / f"practice_{slot}.html").write_text(
            render(template_path.read_text(), manifest, file_sha256(destination), slot)
        )
    shutil.copyfile(guide, output / "먼저읽기.md")
    for slot in ("A", "B"):
        with tarfile.open(output / f"practice_{slot}.tar.gz", "x:gz") as archive:
            for name in ("images", "manifest.json", "먼저읽기.md", f"practice_{slot}.html"):
                archive.add(output / name, arcname=name)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel", type=Path, default=Path("experiments/registration_physical_review_train48_02")
    )
    p.add_argument("--root", type=Path, default=antiuav300_root())
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.panel, a.root, a.output_dir)
    print(f"Prepared {len(result['pairs'])} practice pairs; NOT GT: {a.output_dir}")


if __name__ == "__main__":
    main()
