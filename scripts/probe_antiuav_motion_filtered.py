"""Fixed motion mask filtering of saved matches; annotation-free estimation."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from aero_ir.utils.manifest import file_sha256
from scripts.probe_antiuav_local_warp import corner_iou
from scripts.probe_antiuav_temporal_motion import residual_motion
from scripts.probe_external_local_warp import fit_image_warp
from scripts.verify_antiuav_tiled_matching import safe_artifact, verify


def mask_membership(points, mask, header_top):
    """Nearest pixel-centre sampling after undoing header restoration; no clipping."""
    p = np.asarray(points, dtype=float)
    mask = np.asarray(mask)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("finite Nx2 points required")
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("2D boolean mask required")
    p = p - [0, header_top]
    valid = ((p >= 0) & (p <= np.array(mask.shape[::-1]) - 1)).all(1)
    result = np.zeros(len(p), dtype=bool)
    xy = np.floor(p[valid] + 0.5).astype(int)
    result[valid] = mask[xy[:, 1], xy[:, 0]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    matches_path = Path("experiments/registration_antiuav_tiled_matching_gpu_01/report.json")
    motion_path = Path("experiments/registration_antiuav_temporal_motion_01/report.json")
    expected = {
        matches_path: "1546737338c81b43628435db5a0fb3c3862c18759d08fdb8c5d712f858466733",
        motion_path: "aa22edda051d70c1cf9a0978cb290bdbd366f9265c84607526c0a266e092f0c6",
    }
    for p, digest in expected.items():
        if file_sha256(p) != digest:
            raise ValueError(f"frozen input changed: {p}")
    replay = verify(matches_path)
    matches, motion = [json.loads(p.read_text()) for p in (matches_path, motion_path)]
    hashes = motion["input_and_source_sha256"].copy()
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"motion source changed: {p}")
    panel_path = Path("experiments/xoftr_overlay_train16_01/report.json")
    panel = json.loads(panel_path.read_text())
    cases = {
        (r["sequence_id"], r["frame_index"]): r
        for r in panel["rows"]
        if r["condition"] == "input_header_crop"
    }
    settings_path = Path("configs/experiment/registration_external_local_warp_cpu.yaml")
    settings = yaml.safe_load(settings_path.read_text())
    for p in (
        matches_path,
        motion_path,
        panel_path,
        settings_path,
        Path(__file__),
        Path("scripts/probe_external_local_warp.py"),
        Path("scripts/probe_antiuav_local_warp.py"),
        Path("scripts/verify_antiuav_tiled_matching.py"),
    ):
        hashes[str(p)] = file_sha256(p)
    masks = {}
    for row in motion["rows"]:
        path = safe_artifact(motion_path.parent, row)
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as a:
            score, mask, threshold = residual_motion(a["flows"])
            np.testing.assert_array_equal(mask, a["mask"])
            np.testing.assert_array_equal(score, a["score"])
            if threshold != row["threshold"]:
                raise ValueError("threshold replay differs")
        key = (row["sequence_id"], row["decoded"][0]["frame"], row["modality"])
        if key in masks:
            raise ValueError("duplicate motion input")
        masks[key] = mask
    required = {(seq, frame, m) for seq, frame in cases for m in ("visible", "infrared")}
    if set(masks) != required:
        raise ValueError("incomplete motion panel")
    rows = []
    for row in matches["rows"]:
        seq, frame = row["sequence_id"], row["frame_index"]
        tops = cases[(seq, frame)]["header_rows"]
        pair_masks = [masks[(seq, frame, m)] for m in ("visible", "infrared")]
        for mask, size, top in zip(pair_masks, row["sizes"], tops, strict=True):
            if mask.shape != (size[1] - top, size[0]):
                raise ValueError("crop coordinate contract differs")
        path = safe_artifact(matches_path.parent, row)
        hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as a:
            for variant, prefix in (("baseline", "baseline_"), ("tiled", "")):
                for condition in ("motion", "shifted_control"):
                    used = (
                        pair_masks
                        if condition == "motion"
                        else [
                            np.roll(m, (m.shape[0] // 2, m.shape[1] // 2), axis=(0, 1))
                            for m in pair_masks
                        ]
                    )
                    directions = []
                    for d, names in enumerate(
                        (
                            ("points0", "points1", "confidence"),
                            ("reverse0", "reverse1", "reverse_confidence"),
                        )
                    ):
                        p, q, c = [a[prefix + n] for n in names]
                        source, target = d, 1 - d
                        keep = np.flatnonzero(
                            mask_membership(p, used[source], tops[source])
                            & mask_membership(q, used[target], tops[target])
                        )
                        sizes = row["sizes"] if d == 0 else row["sizes"][::-1]
                        warp, info = fit_image_warp(p[keep], q[keep], c[keep], *sizes, settings)
                        boxes = row["boxes_xyxy"] if d == 0 else row["boxes_xyxy"][::-1]
                        directions.append(
                            dict(kept_indices=keep.tolist(), fit=info, iou=corner_iou(warp, *boxes))
                        )
                    passed = all(x["iou"] is not None and x["iou"] >= 0.6 for x in directions)
                    rows.append(
                        dict(
                            sequence_id=seq,
                            variant=variant,
                            condition=condition,
                            directions=directions,
                            joint_pass=passed,
                        )
                    )
        print(seq, [(r["variant"], r["condition"], r["joint_pass"]) for r in rows[-4:]], flush=True)
    summary = {
        v: {
            c: sum(r["joint_pass"] for r in rows if r["variant"] == v and r["condition"] == c)
            for c in ("motion", "shifted_control")
        }
        for v in ("baseline", "tiled")
    }
    for p, digest in hashes.items():
        if file_sha256(p) != digest:
            raise ValueError(f"changed input: {p}")
    result = dict(
        summary=summary,
        pairs=16,
        rows=rows,
        input_and_source_sha256=hashes,
        source_reconstruction=replay,
        registration_qualified=False,
        generator_training_approved=False,
        limitations=[
            "Train16 box proxy, not dense physical accuracy.",
            "No GT-driven masks, threshold tuning, dilation or extrapolation.",
            "Shifted masks are a location control, not independent wrong-image GT.",
        ],
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
