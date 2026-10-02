"""Train-only automatic lag hypothesis screen using EXISTING box trajectories.

This is not a camera synchronization calibration: independently moving cameras,
annotation noise and nonlinear geometry can confound an estimated lag. It never
changes frame pairing or claims pixel accuracy. No new human annotations needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aero_ir.utils.manifest import file_sha256


def trajectory(annotation, shape):
    present = np.array(annotation["exist"], dtype=bool)
    rows = annotation["gt_rect"]
    if present.shape != (len(rows),):
        raise ValueError("invalid annotation lengths")
    # Anti-UAV encodes some absent frames as [], not a zero-sized rectangle.
    boxes = np.full((len(rows), 4), np.nan)
    for i, row in enumerate(rows):
        if len(row) == 4:
            boxes[i] = row
        elif present[i]:
            raise ValueError("present frame lacks a four-coordinate box")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or present.shape != (len(boxes),):
        raise ValueError("invalid annotation shape")
    valid = present & np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
    centres = (boxes[:, :2] + boxes[:, 2:] / 2) / np.array(shape[::-1])
    return centres, valid


def lag_screen(a, b, valid_a, valid_b, maximum=15):
    if maximum < 1:
        raise ValueError("maximum lag must be positive")
    n = min(len(a), len(b))
    lags = np.arange(-maximum, maximum + 1)
    ids = np.arange(maximum, n - maximum)
    # Every lag uses exactly the same source time indices and existence support.
    keep = valid_a[ids].copy()
    for lag in lags:
        keep &= valid_b[ids + lag]
    ids = ids[keep]
    if len(ids) < 80:
        return {"status": "insufficient_common_frames", "common_frames": len(ids)}
    blocks = np.array_split(ids, 4)
    result = []
    for block in blocks:
        # Forward chronological split inside each block; no fit on check frames.
        cut = len(block) * 2 // 3
        fit, check = block[:cut], block[cut:]
        xfit, xcheck = np.c_[a[fit], np.ones(len(fit))], np.c_[a[check], np.ones(len(check))]
        if np.linalg.matrix_rank(xfit) < 3 or np.linalg.cond(xfit) > 1e6:
            return {"status": "unobservable_degenerate_trajectory", "common_frames": len(ids)}
        train_errors, check_errors = [], []
        for lag in lags:
            matrix, _, _, _ = np.linalg.lstsq(xfit, b[fit + lag], rcond=None)
            train_errors.append(float(np.sqrt(np.mean((xfit @ matrix - b[fit + lag]) ** 2))))
            check_errors.append(float(np.sqrt(np.mean((xcheck @ matrix - b[check + lag]) ** 2))))
        train_errors, check_errors = np.array(train_errors), np.array(check_errors)
        baseline = maximum
        best = (
            baseline
            if train_errors[baseline] <= train_errors.min() + 1e-10
            else int(train_errors.argmin())
        )
        result.append(
            {
                "fit_frames": len(fit),
                "check_frames": len(check),
                "selected_lag": int(lags[best]),
                "fit_rmse_by_lag": train_errors.tolist(),
                "check_rmse_by_lag": check_errors.tolist(),
                "zero_lag_check_rmse": float(check_errors[baseline]),
                "selected_check_rmse": float(check_errors[best]),
                "check_relative_improvement": float(
                    1 - check_errors[best] / max(check_errors[baseline], 1e-12)
                ),
            }
        )
    return {
        "status": "screened_not_calibrated",
        "common_frames": len(ids),
        "lag_definition": "RGB[t] compared with IR[t+lag]",
        "lags": lags.tolist(),
        "blocks": result,
        "all_four_selected_same_lag": len({r["selected_lag"] for r in result}) == 1,
        "all_four_check_improve_at_least_10percent": all(
            r["check_relative_improvement"] >= 0.1 for r in result
        ),
        "any_selected_search_boundary": any(abs(r["selected_lag"]) == maximum for r in result),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300"))
    p.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("experiments/detector_free_train16_01/report.json"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite report")
    baseline = json.loads(args.baseline_report.read_text())
    if baseline["evaluated_split"] != "train" or baseline["validation_or_test_access"] != "none":
        raise ValueError("requires train panel")
    if file_sha256(args.root / "label_new/train.json") != baseline["split_manifest_sha256"]:
        raise ValueError("train split changed")
    rows = []
    for record in baseline["inputs"]:
        inputs, hashes = [], {}
        for modality in ("visible", "infrared"):
            path = args.root / "train" / record["sequence_id"] / f"{modality}.json"
            hashes[modality] = file_sha256(path)
            if hashes[modality] != record["inputs"][modality]["annotation_sha256"]:
                raise ValueError("annotation changed")
            inputs.append(
                trajectory(json.loads(path.read_text()), record["inputs"][modality]["native_shape"])
            )
        (a, va), (b, vb) = inputs
        rows.append(
            {
                "sequence_id": record["sequence_id"],
                "annotation_sha256": hashes,
                "result": lag_screen(a, b, va, vb),
            }
        )
    t = np.linspace(0, 12, 400)
    a = np.c_[np.sin(t) + 0.2 * np.sin(7 * t), np.cos(1.7 * t) + 0.1 * np.sin(11 * t)]
    b = np.roll(a @ np.array([[0.7, 0.1], [-0.2, 0.8]]) + [0.1, 0.2], 4, axis=0)
    valid = np.ones(len(t), bool)
    control = lag_screen(a, b, valid, valid)
    passed = all(x["selected_lag"] == 4 for x in control["blocks"])
    if not passed:
        raise RuntimeError("known-lag control failed")
    candidates = [
        r["sequence_id"]
        for r in rows
        if r["result"].get("all_four_selected_same_lag")
        and r["result"].get("all_four_check_improve_at_least_10percent")
        and not r["result"].get("any_selected_search_boundary")
    ]
    report = {
        "experiment": "train_box_trajectory_lag_screen_v1",
        "evaluated_split": "train",
        "validation_or_test_access": "none",
        "human_input": False,
        "baseline_sha256": file_sha256(args.baseline_report),
        "source_sha256": file_sha256(__file__),
        "rows": rows,
        "known_lag_control": {"expected": 4, "passed": passed},
        "stable_nonboundary_hypotheses": candidates,
        "pairing_changed": False,
        "qualification": "not_assessed_trajectory_proxy_not_camera_calibration",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "sequences": len(rows),
                "stable_nonboundary_hypotheses": candidates,
                "known_lag_control_passed": passed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
