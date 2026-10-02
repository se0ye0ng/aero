"""Prepare and actually load matched-size real/procedural engineering inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.data.coco_mixture import assemble_mixture
from aero_ir.data.preprocess import verify_preprocess_spec
from aero_ir.detect.flir_yolox import build_flir_yolox_dataset
from aero_ir.utils.manifest import file_sha256

ROOT = Path(
    "/lustre/winston1214/dataset/teledyne-flir-adas-thermal-dataset-v2/extracted/FLIR_ADAS_v2"
)
REAL = Path("experiments/flir_yolox/annotations/train.json")
AUX = Path("experiments/flir_procedural_encoding_smoke_01")
PREPROCESS = Path("experiments/flir_preprocess.json")
PINS = {
    REAL: "0d7baab44bda810f25282aefcbe54f19f414243ddce704b4d37a4497d4dab710",
    PREPROCESS: "b5c2f39026eaeadde95d3d9b4dd4c40597cda3d6fced6d4fc7122964fc6824f2",
    AUX / "report.json": "ff7d0a22235f0e1fbc88cfee76e2cc8c291e72f1f367970b662614e5797919f8",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    auxiliary_report = json.loads((AUX / "report.json").read_text())
    hashes = {str(k): v for k, v in PINS.items()}
    hashes.update(auxiliary_report["input_and_source_sha256"])
    hashes[str(AUX / "annotations.json")] = auxiliary_report["annotations_sha256"]
    for row in auxiliary_report["rows"]:
        path = AUX / row["artifact"]
        if not path.resolve().is_relative_to(AUX.resolve()):
            raise ValueError("auxiliary artifact escapes pool root")
        hashes[str(path)] = row["sha256"]
    for path, sha in hashes.items():
        if file_sha256(path) != sha:
            raise ValueError(f"changed input: {path}")
    for path in (
        Path(__file__),
        Path("src/aero_ir/data/coco_mixture.py"),
        Path("src/aero_ir/detect/flir_yolox.py"),
    ):
        hashes[str(path)] = file_sha256(path)
    spec = json.loads(PREPROCESS.read_text())
    verify_preprocess_spec(spec)
    real = json.loads(REAL.read_text())
    auxiliary = json.loads((AUX / "annotations.json").read_text())
    # Paths are bound below to one of the two authorized, hashed source pools.
    for pool, root in ((real, ROOT), (auxiliary, AUX)):
        for image in pool["images"]:
            path = root / image["file_name"]
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("source image escapes authorized root")
            image["file_name"] = str(path.resolve())
    plans = {}
    for arm, real_count, auxiliary_count in (
        ("real_only", 32, 0),
        ("real_plus_procedural", 16, 16),
    ):
        plans[arm] = assemble_mixture(
            real, auxiliary, n_real=real_count, n_auxiliary=auxiliary_count, seed=0
        )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "annotations").mkdir()
    arms = {}
    for arm, (coco, origins) in plans.items():
        coco["info"].update(
            engineering_smoke_only=True,
            aero_manifest_sha256=spec["source_manifest_sha256"],
            manifest_link_role="real-train intensity calibration only",
            pool_input_hashes={
                "real": PINS[REAL],
                "auxiliary": auxiliary_report["annotations_sha256"],
            },
        )
        path = args.out_dir / "annotations" / f"{arm}.json"
        with path.open("x") as f:
            json.dump(coco, f, indent=2, allow_nan=False)
        dataset = build_flir_yolox_dataset(
            image_root=ROOT,
            prepared_root=args.out_dir,
            annotation_file=path.name,
            preprocess_path=PREPROCESS,
        )
        lookup = {image["id"]: image for image in coco["images"]}
        categories = sorted(c["id"] for c in coco["categories"])
        loaded = []
        for index in range(len(dataset)):
            image_id = int(dataset.ids[index])
            original = lookup[image_id]
            source = Path(original["file_name"])
            hashes[str(source)] = file_sha256(source)
            image = dataset.load_image(index)
            if image.dtype != np.uint8 or image.shape != (original["height"], original["width"], 3):
                raise ValueError("loader image representation differs")
            anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
            ratio = min(640 / original["height"], 640 / original["width"])
            expected = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                expected.append(
                    [
                        x * ratio,
                        y * ratio,
                        (x + w) * ratio,
                        (y + h) * ratio,
                        categories.index(ann["category_id"]),
                    ]
                )
            np.testing.assert_allclose(
                dataset.load_anno(index), np.asarray(expected).reshape(-1, 5), rtol=1e-6, atol=1e-5
            )
            loaded.append(dict(image_id=image_id, labels=len(anns), std=float(image.std())))
        arms[arm] = dict(
            images=len(dataset),
            annotations=len(coco["annotations"]),
            annotation_file=str(path.relative_to(args.out_dir)),
            sha256=file_sha256(path),
            origins=origins,
            loaded=loaded,
        )
    for path, sha in hashes.items():
        if file_sha256(path) != sha:
            raise ValueError(f"input changed during smoke: {path}")
    result = dict(
        arms=arms,
        input_and_source_sha256=hashes,
        detector_training_completed=False,
        registration_qualified=False,
        full_experiment_approved=False,
        limitations=[
            "32-image engineering inputs, not the full ratio sweep.",
            "No validation/test images used or AP measured.",
            "Cross-modal scene-disjointness remains unaudited for formal use.",
            "Procedural baseline retains box-conditioned shortcuts.",
        ],
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: {n: v[n] for n in ("images", "annotations")} for k, v in arms.items()}))


if __name__ == "__main__":
    main()
