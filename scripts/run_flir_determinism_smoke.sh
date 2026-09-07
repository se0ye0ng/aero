#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

sanitize_pasted_path() {
  local value="$1"
  local line
  local result=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    result+="$line"
  done <<< "$value"
  printf '%s' "$result"
}

# A previously exported AERO_FLIR_ROOT may contain pasted newlines or indentation.
FLIR_ROOT="$(sanitize_pasted_path "${AERO_FLIR_ROOT:-/lustre/winston1214/dataset/teledyne-flir-adas-thermal-dataset-v2/extracted/FLIR_ADAS_v2}")"
PREPARED_ROOT="${AERO_FLIR_YOLOX_ROOT:-$PROJECT_ROOT/experiments/flir_yolox}"
PREPROCESS_PATH="${AERO_FLIR_PREPROCESS:-$PROJECT_ROOT/experiments/flir_preprocess.json}"
OUTPUT_ROOT="${AERO_YOLOX_OUTPUT:-$PROJECT_ROOT/experiments/yolox_runs}"
RUN_PREFIX="${AERO_DETERMINISM_RUN_PREFIX:-flir_determinism_smoke_seed0}"
REPORT_PATH="${AERO_DETERMINISM_REPORT:-$PROJECT_ROOT/experiments/flir_yolox_determinism_smoke.json}"

if [[ ! "$RUN_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "invalid AERO_DETERMINISM_RUN_PREFIX: $RUN_PREFIX" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi
for required in \
  "$FLIR_ROOT" \
  "$PREPARED_ROOT/annotations/smoke_train.json" \
  "$PREPARED_ROOT/annotations/smoke_val.json" \
  "$PREPROCESS_PATH"; do
  if [[ ! -e "$required" ]]; then
    echo "required input is missing: $required" >&2
    exit 2
  fi
done

RUN_A="$OUTPUT_ROOT/${RUN_PREFIX}_a"
RUN_B="$OUTPUT_ROOT/${RUN_PREFIX}_b"
for run_dir in "$RUN_A" "$RUN_B"; do
  if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite existing run directory: $run_dir" >&2
    echo "Set AERO_DETERMINISM_RUN_PREFIX to a new name to make a fresh attempt." >&2
    exit 2
  fi
done

COMMON_ENV=(
  env -u LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES=0
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0
  AERO_FLIR_ROOT="$FLIR_ROOT"
  AERO_FLIR_YOLOX_ROOT="$PREPARED_ROOT"
  AERO_FLIR_PREPROCESS="$PREPROCESS_PATH"
  AERO_YOLOX_OUTPUT="$OUTPUT_ROOT"
  AERO_YOLOX_MAX_EPOCHS=20
  AERO_YOLOX_EVAL_INTERVAL=20
  AERO_YOLOX_PRINT_INTERVAL=10
  AERO_YOLOX_WORKERS=8
  AERO_YOLOX_SEED=0
  AERO_YOLOX_GRAD_ACCUM=8
  AERO_YOLOX_EFFECTIVE_BATCH=64
  AERO_YOLOX_TRAIN_ANN=smoke_train.json
  AERO_YOLOX_VAL_ANN=smoke_val.json
  AERO_YOLOX_SAVE_HISTORY=0
)

cd "$PROJECT_ROOT"

echo "Checking CUDA before launching either run..."
"${COMMON_ENV[@]}" "$PYTHON_BIN" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)"

run_one() {
  local suffix="$1"
  local run_id="${RUN_PREFIX}_${suffix}"
  local run_dir="$OUTPUT_ROOT/$run_id"
  echo "Starting deterministic smoke $run_id"
  "${COMMON_ENV[@]}" \
    AERO_YOLOX_METRICS_PATH="$run_dir/metrics.json" \
    AERO_YOLOX_PREDICTIONS_PATH="$run_dir/predictions.json" \
    "$PYTHON_BIN" -m yolox.tools.train \
    -f "$PROJECT_ROOT/src/aero_ir/detect/yolox_flir_exp.py" \
    -d 1 \
    -b 8 \
    --fp16 \
    -expn "$run_id" \
    -l tensorboard

  local required_artifact
  for required_artifact in \
    metrics.json \
    predictions.json \
    last_epoch_ckpt.pth \
    runtime.json \
    train_log.txt; do
    if [[ ! -s "$run_dir/$required_artifact" ]]; then
      echo "run $run_id did not complete: missing $run_dir/$required_artifact" >&2
      return 1
    fi
  done
}

run_one a
run_one b

echo "Comparing metrics, predictions, normalized traces, and checkpoint contents..."
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" - "$RUN_A" "$RUN_B" "$REPORT_PATH" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

from aero_ir.utils.manifest import canonical_hash, file_sha256

first = Path(sys.argv[1]).resolve()
second = Path(sys.argv[2]).resolve()
report_path = Path(sys.argv[3]).resolve()


def load_json(run: Path, name: str):
    return json.loads((run / name).read_text(encoding="utf-8"))


def checkpoint_mismatches(left, right, path="checkpoint"):
    mismatches = []
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            mismatches.append(path)
        return mismatches
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [f"{path}.keys"]
        for key in left:
            mismatches.extend(checkpoint_mismatches(left[key], right[key], f"{path}.{key}"))
            if len(mismatches) >= 20:
                break
        return mismatches
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return [f"{path}.length"]
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            mismatches.extend(
                checkpoint_mismatches(left_item, right_item, f"{path}[{index}]")
            )
            if len(mismatches) >= 20:
                break
        return mismatches
    if left != right:
        mismatches.append(path)
    return mismatches


trace_pattern = re.compile(
    r"epoch: (\d+)/20, iter: (\d+)/16,.*?"
    r"total_loss: ([0-9.]+), iou_loss: ([0-9.]+), l1_loss: ([0-9.]+), "
    r"conf_loss: ([0-9.]+), cls_loss: ([0-9.]+), lr: ([0-9.e+-]+), "
    r"size: (\d+),.*?optimizer_step: (\d+)"
)


def normalized_trace(run: Path):
    text = (run / "train_log.txt").read_text(encoding="utf-8", errors="replace")
    return trace_pattern.findall(text)


metric_keys = (
    "status",
    "evaluator",
    "iou_type",
    "images",
    "detections",
    "map_50_95",
    "map_50",
    "mar_50_95",
    "mar_50",
    "per_class",
    "prediction_filter",
)
first_metrics = load_json(first, "metrics.json")
second_metrics = load_json(second, "metrics.json")
metrics_equal = {key: first_metrics[key] for key in metric_keys} == {
    key: second_metrics[key] for key in metric_keys
}
predictions_equal = load_json(first, "predictions.json") == load_json(
    second, "predictions.json"
)
first_checkpoint = torch.load(
    first / "last_epoch_ckpt.pth", map_location="cpu", weights_only=False
)
second_checkpoint = torch.load(
    second / "last_epoch_ckpt.pth", map_location="cpu", weights_only=False
)
checkpoint_differences = checkpoint_mismatches(first_checkpoint, second_checkpoint)
first_trace = normalized_trace(first)
second_trace = normalized_trace(second)
trace_equal = bool(first_trace) and first_trace == second_trace

checks = {
    "scientific_metrics_exact": metrics_equal,
    "predictions_exact": predictions_equal,
    "checkpoint_exact": not checkpoint_differences,
    "normalized_training_trace_exact": trace_equal,
}
report = {
    "schema_version": 1,
    "kind": "flir_yolox_determinism_smoke",
    "gate": "pass" if all(checks.values()) else "hold",
    "runs": [first.name, second.name],
    "checks": checks,
    "logged_trace_records": [len(first_trace), len(second_trace)],
    "checkpoint_mismatches": checkpoint_differences,
    "artifacts": {
        run.name: {
            name: file_sha256(run / name)
            for name in (
                "metrics.json",
                "predictions.json",
                "last_epoch_ckpt.pth",
                "train_log.txt",
            )
        }
        for run in (first, second)
    },
}
report["report_sha256"] = canonical_hash(report)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
print(f"wrote {report_path}")
if report["gate"] != "pass":
    raise SystemExit(1)
PY
