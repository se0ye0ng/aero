"""Validate two independent review files without issuing qualification PASS."""

import argparse
import json
from pathlib import Path

from aero_ir.registration.physical_review import validate_reviews
from aero_ir.utils.manifest import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.panel / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for pair in manifest["pairs"]:
        for item in pair["images"].values():
            path = (args.panel / item["path"]).resolve()
            if not path.is_relative_to(args.panel.resolve()):
                raise ValueError("image path outside panel")
            if file_sha256(path) != item["sha256"]:
                raise ValueError(f"image hash mismatch: {path}")
    reviews = [json.loads(p.read_text()) for p in (args.review_a, args.review_b)]
    result = validate_reviews(manifest, file_sha256(manifest_path), reviews)
    result["review_file_sha256"] = {str(p): file_sha256(p) for p in (args.review_a, args.review_b)}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "reviewer_discrepancies"}, indent=2))
    raise SystemExit(0 if result["structure_valid"] else 2)


if __name__ == "__main__":
    main()
