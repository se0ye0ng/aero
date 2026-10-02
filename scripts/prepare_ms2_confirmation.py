"""Freeze and extract unseen-in-analysis training frames for fixed calibration candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.ms2_archive import extract_screen
from aero_ir.utils.manifest import file_sha256


def interval_midpoints(frames):
    if (len(frames) < 2 or any(not isinstance(f, str) or len(f) != 6 or not f.isdigit()
                              for f in frames)):
        raise ValueError("at least two six-digit frame IDs required")
    values = list(map(int, frames))
    if any(b-a < 2 for a, b in zip(values[:-1], values[1:], strict=True)):
        raise ValueError("strictly ordered endpoints with unseen interior frames required")
    return [f"{(a+b)//2:06d}" for a, b in zip(values[:-1], values[1:], strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "extract"))
    parser.add_argument("--plan", type=Path,
                        default=Path("experiments/ms2_confirmation_plan_01/plan.json"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    base_path = Path("experiments/ms2_train_screen_plan_01/plan.json")
    base_hash = "681cf9c6f3d680cf4d0ee7c3e37575868a56066fa3f1bf19662466dedb2952d0"
    candidate_path = Path("experiments/ms2_stereo_calibration_01/report.json")
    candidate_hash = "c70deb86993fe8c3f1140da3d7f3a35233ea3593cb10b9a3a0b821d61d1cacc4"
    if file_sha256(base_path) != base_hash or file_sha256(candidate_path) != candidate_hash:
        raise ValueError("source selection or candidate changed")
    base = json.loads(base_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if args.action == "plan":
        if args.plan.exists():
            raise FileExistsError(args.plan)
        transforms = {c["name"]: c["transform"] for c in candidate["candidates"]
                      if c["name"] in ("author_calibration", "xoftr_640_rotation_only",
                                       "minima_xoftr_rotation_only")}
        if len(transforms) != 3:
            raise ValueError("missing frozen candidate")
        plan = dict(schema="ms2_train_confirmation_plan_v1", sequence=base["sequence"],
                    frame_ids=interval_midpoints(base["frame_ids"]),
                    frame_count=base["frame_count"],
                    prior_frame_ids=base["frame_ids"], candidate_transforms=transforms,
                    prior_plan_sha256=base_hash, candidate_report_sha256=candidate_hash,
                    selection="integer midpoint of each consecutive frozen16-frame interval",
                    split="official train; new image IDs, not independent sequence or test set",
                    score_thresholds_target_native_px=[1., 3., 5., 10.],
                    refit_allowed=False, frame_exclusion_allowed=False,
                    thresholds_are_qualification_gate=False,
                    confirmation_requires_both_fixed_matchers=True,
                    registration_qualified=False, generator_training_approved=False)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open("x") as handle:
            json.dump(plan, handle, indent=2)
        print(f"wrote {args.plan}; frozen before confirmation image extraction")
        return
    if args.out is None or args.output_root is None:
        parser.error("extract requires --out and --output-root")
    if args.out.exists():
        raise FileExistsError(args.out)
    plan = json.loads(args.plan.read_text())
    if (plan["schema"] != "ms2_train_confirmation_plan_v1"
            or plan["prior_plan_sha256"] != base_hash
            or plan["candidate_report_sha256"] != candidate_hash
            or plan["sequence"] != base["sequence"]
            or plan["frame_ids"] != interval_midpoints(base["frame_ids"])):
        raise ValueError("confirmation plan differs from deterministic protocol")
    archive = Path("experiments/external/ms2_first_train_archives/sync_data.tar.bz2.part")
    expected_archive = "c3a10f2cef0d04ea6999c9885311aa1bda206e3c228960ce3e1462d19e913902"
    if file_sha256(archive) != expected_archive:
        raise ValueError("source archive differs from completed CRC/EOF-verified acquisition")
    result = extract_screen(archive, args.output_root, plan, "sync_data", 24150818939)
    result.update(plan_sha256=file_sha256(args.plan),
                  source_sha256={str(Path(__file__)): file_sha256(__file__),
                                 "src/aero_ir/data/ms2_archive.py": file_sha256(
                                     "src/aero_ir/data/ms2_archive.py")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {args.out}; no qualification or generator approval")
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
