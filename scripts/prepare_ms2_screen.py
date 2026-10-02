"""Create a train-only screen plan, or selectively extract its source assets on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aero_ir.data.ms2_archive import extract_screen, odometry_plan
from aero_ir.utils.manifest import file_sha256
from scripts.audit_ms2_metadata import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "extract"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("experiments/external/ms2_metadata"))
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--kind", choices=("sync_data", "proj_depth"))
    parser.add_argument("--expected-bytes", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.out}")
    source = audit(args.metadata)
    if not source["sequence_id_disjoint"]:
        raise ValueError("official split IDs overlap")
    if args.action == "plan":
        result = odometry_plan(args.archive, source["first_training_sequence"])
    else:
        if any(v is None for v in (args.plan, args.kind, args.expected_bytes, args.output_root)):
            parser.error("extract requires --plan, --kind, --expected-bytes and --output-root")
        plan = json.loads(args.plan.read_text())
        if (plan.get("schema") != "ms2_train_screen_plan_v1"
                or plan["sequence"] != source["first_training_sequence"]):
            raise ValueError("not the predetermined official train sequence")
        result = extract_screen(args.archive, args.output_root, plan,
                                args.kind, args.expected_bytes)
        result["plan_sha256"] = file_sha256(args.plan)
    result["metadata_sha256"] = source["input_sha256"]
    result["source_sha256"] = {
        "prepare_ms2_screen.py": file_sha256(__file__),
        "ms2_archive.py": file_sha256(Path(__file__).resolve().parents[1] /
                                     "src/aero_ir/data/ms2_archive.py"),
        **source["source_sha256"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {args.out}; no registration or generator approval")
    if args.action == "extract" and not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
