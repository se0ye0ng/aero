"""Extend the fixed train-only lag diagnostic to all160 v7 training sequences.

Existing box trajectories and container dimensions only; no pixel inference,
pairing changes, model fitting or registration qualification. Affine trajectory
maps ARE fit inside each diagnostic block, not registration network parameters.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from aero_ir.data.antiuav import load_split_manifest, probe_video
from aero_ir.registration.qualification_v4 import direction_pass
from aero_ir.utils.manifest import canonical_hash, file_sha256
from aero_ir.utils.paths import antiuav300_root
from scripts.probe_registration_timing import lag_screen, trajectory
from scripts.verify_registration_v7 import check, verify_arm


def stable_hypothesis(result):
    blocks = result.get("blocks", [])
    return bool(
        result.get("status") == "screened_not_calibrated"
        and len(blocks) == 4
        and result.get("all_four_selected_same_lag")
        and result.get("all_four_check_improve_at_least_10percent")
        and not result.get("any_selected_search_boundary")
        and blocks[0]["selected_lag"] != 0
    )


def summarize(rows):
    result = {}
    for key in ("actual", "shuffled"):
        strata = {}
        for passed in (False, True):
            selected = [r for r in rows if r["v7_geometry_joint_pass"] == passed]
            stable = [r for r in selected if stable_hypothesis(r[key])]
            strata["v7_pass" if passed else "v7_fail"] = {
                "sequences": len(selected),
                "status_counts": dict(Counter(r[key]["status"] for r in selected)),
                "stable_lag_hypotheses": len(stable),
                "stable_sequence_ids": [r["sequence_id"] for r in stable],
                "stable_lags": {r["sequence_id"]: r[key]["blocks"][0]["selected_lag"]
                                for r in stable},
            }
        result[key] = strata
    return result


def run(root, cache, run_root, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite timing diagnostic")
    project = Path(__file__).resolve().parents[1]
    source_names = (
        "scripts/probe_registration_timing_cohort.py", "scripts/probe_registration_timing.py",
        "src/aero_ir/data/antiuav.py", "scripts/verify_registration_v7.py",
    )
    sources = {name: file_sha256(project / name) for name in source_names}
    proof = verify_arm(run_root / "geometry", "geometry", cache)
    check(proof["status"] == "verified_saved_artifacts_not_replayed", "completed v7 required")
    screen = json.loads((run_root / "geometry/final_train_screen.json").read_text())
    cache_path = cache / "manifest.json"
    cache_hash = file_sha256(cache_path)
    manifest = json.loads(cache_path.read_text())
    train_path = root / "label_new/train.json"
    train_hash = file_sha256(train_path)
    check(train_hash == manifest["split_manifest_sha256"], "official train split changed")
    ids = sorted(load_split_manifest(root, "train"))
    check(len(ids) == 160 and set(ids) == {r["sequence_id"] for r in screen["rows"]},
          "expected the full160-sequence v7 train cohort")
    annotations, videos, trajectories, shard_hashes = {}, {}, {}, {}
    for sid in ids:
        folder = (root / "train" / sid).resolve()
        check(folder.parent == (root / "train").resolve(), "sequence outside train")
        shard_path = cache / "shards" / sid / "manifest.json"
        shard_hashes[sid] = file_sha256(shard_path)
        shard = json.loads(shard_path.read_text())
        expected = next(s["shard_sha256"] for s in manifest["shards"] if s["sequence_id"] == sid)
        check(shard["shard_sha256"] == expected
              and canonical_hash({k: v for k, v in shard.items() if k != "shard_sha256"})
              == expected, "shard identity differs")
        trajectories[sid], videos[sid] = {}, {}
        for modality in ("visible", "infrared"):
            annotation = folder / f"{modality}.json"
            digest = file_sha256(annotation)
            check(digest == shard[f"{modality}_annotations_sha256"], "annotation/cache mismatch")
            annotations[str(annotation.relative_to(root.resolve()))] = digest
            video = folder / f"{modality}.mp4"
            media = probe_video(video)
            check(media["opened"] and min(media["height"], media["width"]) > 0,
                  f"missing native dimensions: {video}")
            videos[sid][modality] = {
                **media, "size_bytes": video.stat().st_size,
                "mtime_ns": video.stat().st_mtime_ns,
            }
            trajectories[sid][modality] = trajectory(json.loads(annotation.read_text()),
                                                      [media["height"], media["width"]])
    gates = {r["sequence_id"]: all(direction_pass(r[k])
                                 for k in ("ir_to_rgb_points", "rgb_to_ir_points"))
             for r in screen["rows"]}
    rows = []
    for i, sid in enumerate(ids):
        other = ids[(i + 1) % len(ids)]
        a, va = trajectories[sid]["visible"]
        b, vb = trajectories[sid]["infrared"]
        negative, vn = trajectories[other]["infrared"]
        rows.append({"sequence_id": sid, "v7_geometry_joint_pass": gates[sid],
                     "shuffled_ir_sequence": other,
                     "actual": lag_screen(a, b, va, vb),
                     "shuffled": lag_screen(a, negative, va, vn)})
    for name, digest in sources.items():
        check(file_sha256(project / name) == digest, "diagnostic source drift")
    check(file_sha256(cache_path) == cache_hash, "cache manifest drift")
    check(file_sha256(train_path) == train_hash, "train split drift")
    for name, digest in annotations.items():
        check(file_sha256(root / name) == digest, "annotation drift")
    for sid, digest in shard_hashes.items():
        check(file_sha256(cache / "shards" / sid / "manifest.json") == digest, "shard drift")
        for modality, info in videos[sid].items():
            stat = (root / "train" / sid / f"{modality}.mp4").stat()
            check(stat.st_size == info["size_bytes"] and stat.st_mtime_ns == info["mtime_ns"],
                  "video metadata source changed")
    for name, digest in proof["artifacts_sha256"].items():
        check(file_sha256(run_root / "geometry" / name) == digest, "v7 evidence drift")
    report = {
        "kind": "v7_train_cohort_box_trajectory_lag_diagnostic",
        "evaluated_split": "train", "validation_or_test_access": "none",
        "v7_evidence_sha256": proof["artifacts_sha256"], "source_sha256": sources,
        "official_train_sha256": train_hash, "cache_manifest_sha256": cache_hash,
        "annotation_sha256": annotations, "native_video_metadata": videos,
        "protocol": {"maximum_lag": 15, "chronological_blocks": 4,
                     "fit_fraction": "first2/3 of each block", "check_fraction": "last1/3",
                     "selection": "minimize fit RMSE only; preserve zero on tie",
                     "common_support": "identical valid source indices across all31 lags",
                     "stable_hypothesis": "same nonzero nonboundary lag in all4 blocks; "
                     "each check RMSE improves at least10%",
                     "negative": "cyclic next lexicographic train sequence IR; "
                     "no refitting policy change"},
        "summary": summarize(rows), "rows": rows, "pairing_changed": False,
        "generator_training_eligible": "hold_not_qualified",
        "limitations": [
            "Existing box trajectories, not pixel correspondences or camera calibration.",
            "Tracking, parallax, annotation noise and affine approximation confound inferred lags.",
            "Negative and actual pairs can have different common-support sample counts.",
            "Adjacent IDs may share a recording; shuffled controls are not proven unrelated.",
            "Association with one-frame v7 train pass is not causal or held-out evidence.",
            "Video container metadata read only; video bytes not hashed or frames decoded.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=antiuav300_root())
    parser.add_argument("--cache-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    parser.add_argument("--run-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v7_pilot_e10_seed0"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.cache_root, args.run_root, args.out)


if __name__ == "__main__":
    main()
