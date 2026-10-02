"""Verify and summarize MS2 image-only warps and approximate raw-match references."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from aero_ir.utils.manifest import file_sha256


def direct_reference_scores(points, matched, reference):
    """Nearest source-depth anchor comparison, not exact subpixel image GT.

    Search all source anchors once, then apply masks. Never search again only
    among favorable/depth-consistent anchors. Report source-distance slices.
    """
    points, matched = np.asarray(points), np.asarray(matched)
    if (points.ndim != 2 or points.shape[1] != 2 or points.shape != matched.shape
            or not np.isfinite(points).all() or not np.isfinite(matched).all()):
        raise ValueError("finite corresponding Nx2 image match arrays required")
    result = {"image_matches": len(points), "slices": {}}
    if not len(reference["source_xy"]) or not len(points):
        return result
    distance, indices = cKDTree(reference["source_xy"]).query(points)
    errors = np.linalg.norm(matched - reference["target_xy"][indices], axis=1)
    for mask in ("geometric_mask", "zbuffer_mask", "depth_consistency_mask"):
        result["slices"][mask] = {}
        for radius in (.25, .5, 1.):
            eligible = (distance <= radius) & reference[mask][indices]
            error = errors[eligible]
            result["slices"][mask][str(radius)] = dict(
                associated_matches=len(error),
                target_native_median_px=float(np.median(error)) if len(error) else None,
                within_target_native_px_counts={str(t): int(np.count_nonzero(error <= t))
                                                for t in (1., 3., 5., 10.)})
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for variant, scores in row["scores"].items():
            for mask, item in scores["masks"].items():
                groups[(row["model"], row["condition"], row["source"], variant, mask)].append(item)
    output = []
    for key, items in sorted(groups.items()):
        references = sum(item["references"] for item in items)
        usable = [item for item in items if item["references"]]
        stats = {}
        for threshold in ("1.0", "3.0", "5.0", "10.0"):
            fractions = [item["fraction_within_target_native_px"][threshold] for item in usable]
            weighted = sum(item["fraction_within_target_native_px"][threshold]*item["references"]
                           for item in usable)
            stats[threshold] = dict(equal_frame_fraction=float(np.mean(fractions))
                                    if fractions else None,
                                    pooled_reference_fraction=weighted/references
                                    if references else None)
        medians = [item["conditional_finite_median_px"] for item in items
                   if item["conditional_finite_median_px"] is not None]
        output.append(dict(zip(("model", "condition", "source", "reference", "mask"),
                               key, strict=True)) | dict(
            frames=len(items), frames_without_references=len(items)-len(usable),
            references=references,
            unsupported_predictions=sum(item["unsupported_predictions"] for item in items),
            median_of_conditional_frame_medians_px=float(np.median(medians)) if medians else None,
            threshold_summary=stats))
    return output


def analyze(report_path: Path):
    result = json.loads(report_path.read_text())
    if result.get("schema") != "ms2_image_matching_diagnostic_v1":
        raise ValueError("unexpected source report")
    root = report_path.parent
    preflight_path = root / "preflight.json"
    if file_sha256(preflight_path) != result["preflight_sha256"]:
        raise ValueError("preflight changed")
    preflight = json.loads(preflight_path.read_text())
    for path, digest in result["input_and_source_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"changed dependency: {path}")
    refs, hashes = {}, {}
    for item in preflight["reference_files"]:
        path = root / item["path"]
        if path.parent != root or file_sha256(path) != item["sha256"]:
            raise ValueError("reference file changed or unsafe path")
        with np.load(path, allow_pickle=False) as arrays:
            refs[path.stem] = dict(arrays)
        hashes[str(path)] = item["sha256"]
    seen, direct = set(), []
    for row in result["rows"]:
        key = row["model"], row["frame"], row["condition"], row["source"]
        if key in seen or row["frame"] not in preflight["frame_ids"]:
            raise ValueError("duplicate or unexpected case")
        seen.add(key)
        path = root / row["matches_file"]
        if path.parent != root or file_sha256(path) != row["matches_sha256"]:
            raise ValueError("match file changed or unsafe path")
        hashes[str(path)] = row["matches_sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            scores = {variant: direct_reference_scores(
                arrays["source_xy"], arrays["target_xy"],
                refs[f"reference_{row['frame']}_{row['source']}_{variant}"])
                for variant in ("static", "ego")}
        direct.append(dict(model=row["model"], frame=row["frame"], condition=row["condition"],
                           source=row["source"], scores=scores))
    expected = {(model, frame, condition, source) for model in ("xoftr_640", "minima_xoftr")
                for frame in preflight["frame_ids"] for condition, source in
                (("paired", "rgb"), ("paired", "thr"), ("unrelated_thermal", "rgb"))}
    if seen != expected:
        raise ValueError("incomplete planned comparison; do not silently analyze a subset")
    return dict(schema="ms2_image_matching_analysis_v1", aggregate=summarize(result["rows"]),
                direct_match_rows=direct, controls=result["controls"],
                artifact_sha256=hashes, report_sha256=file_sha256(report_path),
                preflight_sha256=file_sha256(preflight_path),
                direct_reference_note="nearest source anchor approximation, not subpixel image GT",
                registration_qualified=False, generator_training_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = analyze(args.report)
    result["analysis_source_sha256"] = file_sha256(__file__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"wrote {args.out}; no registration qualification")


if __name__ == "__main__":
    main()
