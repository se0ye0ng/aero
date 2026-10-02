"""Train16 motion-localization feasibility, not cross-modal registration accuracy."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.utils.manifest import file_sha256
from aero_ir.utils.paths import antiuav300_root
from scripts.audit_antiuav300_dense_registration import _read_at
from scripts.probe_registration_rgb_resolution import prepare_input


def residual_motion(flows):
    """Subtract global median translation; require motion in both temporal directions."""
    f = np.asarray(flows, dtype=float)
    if f.ndim != 4 or f.shape[0] != 2 or f.shape[-1] != 2 or not np.isfinite(f).all():
        raise ValueError("two finite HxWx2 flows required")
    translation = np.median(f, axis=(1, 2), keepdims=True)
    score = np.min(np.linalg.norm(f - translation, axis=-1), axis=0)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    threshold = max(0.5, median + 3 * 1.4826 * mad)
    return score, score > threshold, threshold


def box_mask(shape, box):
    yy, xx = np.indices(shape)
    x0, y0, x1, y1 = box
    return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=antiuav300_root()
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    cv2.setNumThreads(1)
    source_path = Path("experiments/xoftr_overlay_train16_01/report.json")
    if (
        file_sha256(source_path)
        != "09081366f56d022c7414b83b416d81837bafd37435e88bf6e554d9df2ecd18ef"
    ):
        raise ValueError("panel changed")
    source = json.loads(source_path.read_text())
    if source["evaluated_split"] != "train" or source["validation_or_test_access"] != "none":
        raise ValueError("train only")
    cases = [r for r in source["rows"] if r["condition"] == "input_header_crop"]
    if len(cases) != 16 or len({r["sequence_id"] for r in cases}) != 16:
        raise ValueError("incomplete panel")
    metadata = {r["sequence_id"]: r["inputs"] for r in source["inputs"]}
    files = [
        source_path,
        Path(__file__),
        Path("scripts/probe_registration_rgb_resolution.py"),
        Path("scripts/audit_antiuav300_dense_registration.py"),
    ]
    hashes = {str(p): file_sha256(p) for p in files}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, case in enumerate(cases):
        for m, name in enumerate(("visible", "infrared")):
            seq, frame = case["sequence_id"], case["frame_index"]
            path = args.root / "train" / seq / f"{name}.mp4"
            cap = cv2.VideoCapture(str(path))
            entry = metadata[seq][name]
            images, decoded = [], []
            try:
                for offset in (0, -5, 5):
                    native = _read_at(cap, frame + offset, path)
                    digest = hashlib.sha256(native.tobytes()).hexdigest()
                    if offset == 0 and digest != entry["decoded_native_sha256"]:
                        raise ValueError("central frame changed")
                    gray, _ = prepare_input(
                        native, entry["resized_shape"], case["header_rows"][m], 640
                    )
                    images.append(gray)
                    decoded.append(dict(frame=frame + offset, native_sha256=digest))
            finally:
                cap.release()
            flows = [
                cv2.calcOpticalFlowFarneback(images[0], image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                for image in images[1:]
            ]
            score, mask, threshold = residual_motion(flows)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            # Largest integrated motion score, with fixed minimum area; no annotation selection.
            candidates = [j for j in range(1, count) if stats[j, cv2.CC_STAT_AREA] >= 4]
            best = max(candidates, key=lambda j: float(score[labels == j].sum()), default=0)
            component = labels == best if best else np.zeros_like(mask)
            # Evaluate only after unsupervised masks and component ranking are fixed.
            box = np.array(case["boxes_xyxy"][m]) - [
                0,
                case["header_rows"][m],
                0,
                case["header_rows"][m],
            ]
            target = box_mask(mask.shape, box)
            if not target.any():
                raise ValueError("target outside cropped image")
            intersection = int((component & target).sum())
            union = int((component | target).sum())
            artifact = args.out_dir / f"{i:03d}_{name}.npz"
            np.savez_compressed(
                artifact,
                center=images[0],
                previous=images[1],
                next=images[2],
                flows=np.asarray(flows),
                score=score,
                mask=mask,
                selected_component=component,
            )
            rows.append(
                dict(
                    sequence_id=seq,
                    modality=name,
                    decoded=decoded,
                    artifact=artifact.name,
                    sha256=file_sha256(artifact),
                    threshold=threshold,
                    image_mask_fraction=float(mask.mean()),
                    target_mask_coverage=float(mask[target].mean()),
                    selected_component_box_iou=intersection / union,
                    selected_component_area=int(component.sum()),
                )
            )
            print(seq, name, rows[-1]["target_mask_coverage"], intersection / union, flush=True)
    summary = {
        name: dict(
            images=16,
            median_target_coverage=float(
                np.median([r["target_mask_coverage"] for r in rows if r["modality"] == name])
            ),
            median_image_fraction=float(
                np.median([r["image_mask_fraction"] for r in rows if r["modality"] == name])
            ),
            median_selected_box_iou=float(
                np.median([r["selected_component_box_iou"] for r in rows if r["modality"] == name])
            ),
        )
        for name in ("visible", "infrared")
    }
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed input: {path}")
    result = dict(
        rows=rows,
        summary=summary,
        input_and_source_sha256=hashes,
        opencv_version=cv2.__version__,
        frame_offsets=[-5, 5],
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Motion localization only, not cross-modal correspondence.",
            "Median translation does not model camera rotation or parallax.",
            "Box coverage includes background and is not segmentation accuracy.",
        ],
    )
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
