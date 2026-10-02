"""CPU initialization diagnostic, not training: can v6 displacements seed an SVF?

A displacement is not the logarithm of a deformation. This deliberately tests
the naive warm start before spending GPU training time; it must not be presented
as a trained diffeomorphic matcher or as preservation of the original point map.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from aero_ir.registration.geometry import from_superfusion
from aero_ir.registration.qualification_v4 import direction_statistics
from aero_ir.registration.shared_velocity import shared_fields
from aero_ir.registration.superfusion import load_superfusion_matcher
from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.compare_antiuav300_registration_pilots import DIRECTIONS, checkpoint_info, summary_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-report",
        type=Path,
        default=Path("experiments/registration_inverse_probe_v6_train16_01/report.json"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/antiuav300_registration_v2_full_train_cache"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose fresh output directory")
    parent = json.loads(args.parent_report.read_text())
    if parent.get("kind") != "train_only_inverse_feasibility_probe":
        parser.error("requires previous inverse diagnostic with frozen sample identities")
    usage = parent["data_usage"]
    if usage["fit_split"] != "train" or usage["validation_or_test_access"] != "none":
        parser.error("train-only inputs required")
    recorded = parent["checkpoint"]
    checkpoint = Path(recorded["path"])
    torch.set_num_threads(1)
    info = checkpoint_info(checkpoint)
    if info["sha256"] != recorded["sha256"]:
        raise ValueError("parent checkpoint changed")
    manifest_path = args.cache_root / "manifest.json"
    if file_sha256(manifest_path) != parent["cache_manifest_sha256"]:
        raise ValueError("parent cache changed")
    manifest = json.loads(manifest_path.read_text())
    unsigned = {k: v for k, v in manifest.items() if k != "cache_manifest_sha256"}
    if (
        canonical_hash(unsigned) != manifest["cache_manifest_sha256"]
        or manifest["cache_manifest_sha256"] != info["training_metadata"]["cache_manifest_sha256"]
        or manifest["fit_split"] != "train"
        or manifest["validation_or_test_access"] != "none"
    ):
        raise ValueError("invalid train-cache provenance")
    samples = usage["samples"]
    keys = [(r["sequence_id"], r["cache_position"]) for r in samples]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("empty or duplicated sample selection")
    project = Path(__file__).resolve().parents[1]
    names = [
        "scripts/probe_registration_shared_velocity.py",
        "scripts/compare_antiuav300_registration_pilots.py",
        "src/aero_ir/registration/shared_velocity.py",
        "src/aero_ir/registration/geometry.py",
        "src/aero_ir/registration/qualification_v4.py",
        "src/aero_ir/registration/superfusion.py",
    ]
    sources = {s: file_sha256(project / s) for s in names}
    parent_digest = file_sha256(args.parent_report)
    model = load_superfusion_matcher(checkpoint, torch.device("cpu")).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name in names:
        target = args.output_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
        artifacts[str(target.relative_to(args.output_dir))] = file_sha256(target)
    rows = {k: [] for k in ("learned_pair", "shared_velocity_naive_initialization")}
    started = time.monotonic()
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            folder = (args.cache_root / "shards" / sample["sequence_id"]).resolve()
            if not folder.is_relative_to((args.cache_root / "shards").resolve()):
                raise ValueError("sample path outside cache")
            arrays = {}
            for key, digest in sample["selected_array_sha256"].items():
                if key not in {"visible", "infrared", "source_boxes", "target_boxes"}:
                    raise ValueError("unknown cache array")
                value = np.array(
                    np.load(folder / f"{key}.npy", mmap_mode="r")[sample["cache_position"]]
                )
                if hashlib.sha256(value.tobytes()).hexdigest() != digest:
                    raise ValueError("selected frame content changed")
                arrays[key] = torch.from_numpy(value)
            vis, ir = [
                arrays[k].permute(2, 0, 1)[None].float() / 255 for k in ("visible", "infrared")
            ]
            vb, ib = [arrays[k][None] for k in ("source_boxes", "target_boxes")]
            forward = from_superfusion(model(ir, vis, direction="visible_to_infrared"))
            reverse = from_superfusion(model(ir, vis, direction="infrared_to_visible"))
            variants = {
                "learned_pair": (forward, reverse),
                "shared_velocity_naive_initialization": shared_fields(forward, steps=7),
            }
            for name, (first, second) in variants.items():
                a = direction_statistics(first, second, ib, vb)[0]
                b = direction_statistics(second, first, vb, ib)[0]
                for row in (a, b):
                    row["global_nonpositive_jacobian_fraction"] = (
                        1 - row["positive_jacobian_fraction"]
                    )
                rows[name].append(
                    {
                        "sequence_id": sample["sequence_id"],
                        "frame_index": sample["cache_position"],
                        "index_semantics": "cache_position_not_video_frame",
                        DIRECTIONS[0]: a,
                        DIRECTIONS[1]: b,
                    }
                )
            print(f"{i + 1}/{len(samples)} shared-velocity initialization check", flush=True)
    if (
        file_sha256(args.parent_report) != parent_digest
        or file_sha256(checkpoint) != info["sha256"]
        or file_sha256(manifest_path) != parent["cache_manifest_sha256"]
        or any(file_sha256(project / s) != h for s, h in sources.items())
    ):
        raise ValueError("source/input changed during execution")
    report = {
        "kind": "shared_velocity_naive_initialization_train_diagnostic",
        "parent_report_sha256": parent_digest,
        "checkpoint": info,
        "source_sha256": sources,
        "artifacts_sha256": artifacts,
        "data_usage": usage,
        "protocol": {
            "integration_steps": 7,
            "optimizer_steps": 0,
            "displacement_is_not_log_deformation": True,
            "annotations_enter_transform": False,
            "reverse_from_same_velocity": True,
            "interpolation_does_not_guarantee_exact_inverse": True,
        },
        "runtime": {"device": "cpu", "seconds": time.monotonic() - started},
        "generator_training_eligible": "hold_not_qualified",
        "variants": {
            name: {"summary": summary_for(value), "rows": value} for name, value in rows.items()
        },
    }
    with (args.output_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for name, item in report["variants"].items():
        print(name, "joint_pass", item["summary"]["joint_frame_pass_rate"], flush=True)


if __name__ == "__main__":
    main()
