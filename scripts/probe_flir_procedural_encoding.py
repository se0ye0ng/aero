"""Native RGB-label procedural smoke with a frozen train-only DN encoding."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from aero_ir.data.preprocess import apply_uint16_linear_preprocess, verify_preprocess_spec
from aero_ir.generate.dn_encoding import encode_procedural, procedural_encoding
from aero_ir.generate.synthetic_baseline import SyntheticBaselineGenerator
from aero_ir.utils.manifest import file_sha256
from aero_ir.utils.paths import flir_root

ROOT = flir_root()
CONFIG = Path("configs/generator/synthetic_baseline.yaml")
PREPROCESS = Path("experiments/flir_preprocess.json")
CLASSES = ("person", "bike", "car", "motor", "bus", "truck")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    annotation_path = ROOT / "images_rgb_train/coco.json"
    hashes = {
        str(path): file_sha256(path)
        for path in (
            annotation_path,
            CONFIG,
            PREPROCESS,
            Path(__file__),
            Path("src/aero_ir/generate/synthetic_baseline.py"),
            Path("src/aero_ir/generate/dn_encoding.py"),
            Path("src/aero_ir/data/preprocess.py"),
            Path("src/aero_ir/utils/manifest.py"),
        )
    }
    annotations = json.loads(annotation_path.read_text())
    preprocess = json.loads(PREPROCESS.read_text())
    verify_preprocess_spec(preprocess)
    config = yaml.safe_load(CONFIG.read_text())
    if (
        config.pop("name") != "synthetic_baseline"
        or config.pop("_target_")
        != "aero_ir.generate.synthetic_baseline.SyntheticBaselineGenerator"
    ):
        raise ValueError("unexpected procedural implementation")
    generator = SyntheticBaselineGenerator(**config)
    encoding = procedural_encoding(generator, preprocess)
    names = {c["id"]: c["name"] for c in annotations["categories"] if c["name"] in CLASSES}
    if set(names.values()) != set(CLASSES):
        raise ValueError("class inventory differs")
    by_image = defaultdict(list)
    for ann in annotations["annotations"]:
        if ann["category_id"] in names:
            by_image[ann["image_id"]].append(ann)
    inventory = sorted(annotations["images"], key=lambda r: r["id"])
    indices = sorted(np.random.default_rng(0).choice(len(inventory), 16, replace=False))
    selected = [inventory[i] for i in indices]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "images").mkdir()
    rows, images, output_annotations = [], [], []
    for item in selected:
        path = ROOT / "images_rgb_train" / item["file_name"]
        if not path.resolve().is_relative_to((ROOT / "images_rgb_train").resolve()):
            raise ValueError("RGB image outside training directory")
        hashes[str(path)] = file_sha256(path)
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        if rgb.shape[:2] != (item["height"], item["width"]):
            raise ValueError("native image and annotations use different dimensions")
        anns = sorted(by_image[item["id"]], key=lambda a: a["id"])
        boxes = [a["bbox"] for a in anns]
        labels = [names[a["category_id"]] for a in anns]
        raw = generator.render(rgb, boxes, labels)
        encoded = encode_procedural(raw, encoding, preprocess)
        before = apply_uint16_linear_preprocess(np.rint(raw).astype(np.uint16), preprocess)
        after = apply_uint16_linear_preprocess(encoded, preprocess)
        artifact = args.out_dir / "images" / f"{item['id']:06d}.tiff"
        Image.fromarray(encoded).save(artifact)
        with Image.open(artifact) as saved:
            np.testing.assert_array_equal(np.asarray(saved), encoded)
        rows.append(
            dict(
                image_id=item["id"],
                annotations=len(anns),
                shape=list(raw.shape),
                raw_min=float(raw.min()),
                raw_max=float(raw.max()),
                before_black_fraction=float((before[:, :, 0] == 0).mean()),
                after_black_fraction=float((after[:, :, 0] == 0).mean()),
                after_white_fraction=float((after[:, :, 0] == 255).mean()),
                after_std=float(after[:, :, 0].std()),
                artifact=str(artifact.relative_to(args.out_dir)),
                sha256=file_sha256(artifact),
            )
        )
        images.append({**item, "file_name": str(artifact.relative_to(args.out_dir))})
        output_annotations.extend(anns)
        print(
            f"image {item['id']}: before black {rows[-1]['before_black_fraction']:.4f}; "
            f"after std {rows[-1]['after_std']:.3f}",
            flush=True,
        )
    coco = dict(
        images=images,
        annotations=output_annotations,
        categories=[c for c in annotations["categories"] if c["id"] in names],
        info=dict(
            engineering_smoke_only=True,
            source_split="images_rgb_train",
            input_coordinates="native RGB; no RGB-to-IR label transfer",
            encoding_sha256=encoding["encoding_sha256"],
        ),
    )
    coco_path = args.out_dir / "annotations.json"
    with coco_path.open("x") as f:
        json.dump(coco, f, indent=2, allow_nan=False)
    for path, sha in hashes.items():
        if file_sha256(path) != sha:
            raise ValueError(f"input changed during run: {path}")
    result = dict(
        rows=rows,
        encoding=encoding,
        input_and_source_sha256=hashes,
        annotations_sha256=file_sha256(coco_path),
        selection_seed=0,
        source_split="images_rgb_train",
        registration_qualified=False,
        full_experiment_approved=False,
        limitations=[
            "Numerical encoding, not physical calibration or thermal realism.",
            "Native RGB boxes retained; no cross-modal transfer used.",
            "Procedural box-shaped intensity imprint may create shortcuts.",
            "16-image engineering smoke; no detector training or AP claim.",
        ],
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
