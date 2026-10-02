"""Read-only comparison of completed continuation/residual pilots, never GO."""

import argparse
import json
from pathlib import Path

from aero_ir.utils.manifest import canonical_hash, file_sha256
from scripts.train_registration_residual_pilot import ARMS, KIND, LEARNING_RATES, verify_products


def compare(root, cache):
    specs, products = {}, {}
    for arm in ARMS:
        folder = root / arm
        spec = json.loads((folder / "run_spec.json").read_text())
        if (spec["arm"] != arm or spec["epochs"] != 10 or spec["batch_size"] != 8
                or spec["seed"] != 0 or spec["smoke_steps"] != 0
                or spec["kind"] != KIND or spec["learning_rate"] != LEARNING_RATES[arm]):
            raise ValueError("not the completed frozen pilot protocol")
        products[arm] = verify_products(folder, cache, spec)
        if json.loads((folder / "training_result.json").read_text()) != products[arm]:
            raise ValueError("completion record differs")
        specs[arm] = spec
    a, b = (specs[arm] for arm in ARMS)
    differences = {"arm", "architecture", "learning_rate"}
    if ({k: v for k, v in a.items() if k not in differences}
            != {k: v for k, v in b.items() if k not in differences}):
        raise ValueError("unmatched specification")
    left, right = (products[arm] for arm in ARMS)
    if (left["trace"]["consumed_data_signature"] != right["trace"]["consumed_data_signature"]
            or left["screen_sample_signature"] != right["screen_sample_signature"]):
        raise ValueError("different consumed training data or screen observations")
    return {"kind": "matched_continuation_residual_saved_pilot_comparison",
            "matched_budget_and_data": True,
            "learning_rates": {arm: specs[arm]["learning_rate"] for arm in ARMS},
            "initial_checkpoint_sha256": a["initial_checkpoint_sha256"],
            "arms": products,
            "residual_minus_continuation_joint_pass_rate":
                right["final"]["joint_frame_pass_rate"] - left["final"]["joint_frame_pass_rate"],
            "run_spec_sha256": {arm: canonical_hash(specs[arm]) for arm in ARMS},
            "model_inference_replayed": False, "cache_array_bytes_rehashed": False,
            "generator_training_eligible": "hold_not_qualified",
            "limitations": ["Train midpoint screen, not independent physical accuracy.",
                            "Equal update budget, different trainable capacity and learning rates.",
                            "Not bitwise replay or the final300-epoch experiment."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path("experiments/registration_residual_pilot_e10_seed0"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("experiments/antiuav300_registration_v2_full_train_cache"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run_root, args.cache_root)
    result["comparison_source_sha256"] = file_sha256(__file__)
    if args.out.exists():
        if json.loads(args.out.read_text()) != result:
            parser.error("existing comparison differs; choose a fresh output, no overwrite")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": "saved_evidence_verified_not_qualified",
                      "joint_pass": {arm: r["final"]["joint_frame_pass_rate"]
                                     for arm, r in result["arms"].items()},
                      "report": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
