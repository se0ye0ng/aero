"""Extract disjoint quarter-panel LiDAR without changing the frozen sampling candidate."""

import json
from pathlib import Path

from aero_ir.data.ms2_archive import extract_screen
from aero_ir.utils.manifest import file_sha256
from scripts.prepare_ms2_intrinsic_confirmation import PLAN_PATH, checked_plan

OUT = Path("experiments/ms2_vertical_confirmation_depth_01")
SOURCE = Path("experiments/ms2_vertical_compensation_01/report.json")
SOURCE_SHA = "e480d336f6320f9968c44eec2e2aa24aede101e6a0a4ee920236a93ca14c8afb"
ARCHIVE = Path("experiments/external/ms2_first_train_archives/proj_depth.tar.bz2.part")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("candidate evidence changed")
    plan = checked_plan(PLAN_PATH)
    if file_sha256(ARCHIVE) != "b148bda7f56ab0370bbb8dfbad05fd1a4b699b083186a4354dc9993e6a60f026":
        raise ValueError("depth archive changed")
    evidence = json.loads(SOURCE.read_text())
    if set(plan["frame_ids"]) & {r["frame"] for r in evidence["rows"]}:
        raise ValueError("confirmation overlaps candidate-selection frames")
    protocol = dict(
        frame_ids=plan["frame_ids"],
        sequence=plan["sequence"],
        offsets=[0.0, 0.5, -0.5],
        wrong_pair_cyclic_offset=7,
        camera_fit=False,
        thermal_window_dn=[3308.0, 4974.0],
        selection="pre-existing quarter panel, disjoint from offset-selection panel",
        limitation=(
            "Previously used for other diagnostics; same sequence, not blind or independent scene"
        ),
        source_sha256=SOURCE_SHA,
        plan_sha256=file_sha256(PLAN_PATH),
        preparation_source_sha256=file_sha256(__file__),
        registration_qualified=False,
        generator_training_approved=False,
    )
    plan_path = OUT.with_name(OUT.name + "_plan.json")
    with plan_path.open("x") as f:
        json.dump(protocol, f, indent=2, allow_nan=False)
    print(f"Frozen plan: {plan_path}; extracting 30 original single-scan depth maps", flush=True)
    result = extract_screen(ARCHIVE, OUT, plan, "proj_depth", 19751824021)
    result["plan_sha256"] = file_sha256(plan_path)
    with OUT.with_name(OUT.name + "_report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    if not result["complete"]:
        raise RuntimeError("missing required depth maps")
    print("Extraction verified; no qualification approval", flush=True)


if __name__ == "__main__":
    main()
