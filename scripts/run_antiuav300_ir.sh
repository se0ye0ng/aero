#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
PREPARED_ROOT="${AERO_ANTIUAV300_IR_PREPARED:-$PROJECT_ROOT/experiments/antiuav300_ir_yolox}"
OUTPUT_ROOT="${AERO_YOLOX_OUTPUT:-$PROJECT_ROOT/experiments/yolox_runs}"
RUN_ID="${AERO_ANTIUAV300_RUN_ID:-antiuav300_native_ir_smoke_seed0}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
EPOCHS="${AERO_ANTIUAV300_EPOCHS:-1}"
PRINT_INTERVAL="${AERO_ANTIUAV300_PRINT_INTERVAL:-5}"
TRAINING_PROFILE="${AERO_ANTIUAV300_TRAINING_PROFILE:-}"
if [[ -z "$TRAINING_PROFILE" ]]; then
  if [[ "$EPOCHS" == "1" ]]; then
    TRAINING_PROFILE="smoke"
  else
    TRAINING_PROFILE="standard"
  fi
fi
if [[ "$TRAINING_PROFILE" == "smoke" ]]; then
  DEFAULT_EVAL_INTERVAL=1
else
  DEFAULT_EVAL_INTERVAL=10
fi
EVAL_INTERVAL="${AERO_ANTIUAV300_EVAL_INTERVAL:-$DEFAULT_EVAL_INTERVAL}"
CUDA_DEVICE_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "invalid AERO_ANTIUAV300_RUN_ID: $RUN_ID" >&2
  exit 2
fi
for numeric_setting in "$EPOCHS" "$EVAL_INTERVAL" "$PRINT_INTERVAL"; do
  if [[ ! "$numeric_setting" =~ ^[1-9][0-9]*$ ]]; then
    echo "epochs, evaluation interval and print interval must be positive integers" >&2
    exit 2
  fi
done
if [[ "$TRAINING_PROFILE" == "smoke" && "$EPOCHS" != "1" ]]; then
  echo "the smoke profile requires exactly one epoch" >&2
  exit 2
fi
if [[ "$TRAINING_PROFILE" == "standard" && "$EPOCHS" -le 20 ]]; then
  echo "the standard profile requires more than 20 epochs" >&2
  exit 2
fi
if [[ "$TRAINING_PROFILE" != "smoke" && "$TRAINING_PROFILE" != "standard" ]]; then
  echo "AERO_ANTIUAV300_TRAINING_PROFILE must be smoke or standard" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite existing run directory: $RUN_DIR" >&2
  echo "Set AERO_ANTIUAV300_RUN_ID to a fresh name." >&2
  exit 2
fi
if [[ "$TRAINING_PROFILE" == "standard" && -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
  echo "standard training requires a clean Git checkout" >&2
  git -C "$PROJECT_ROOT" status --short >&2
  exit 2
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Verifying every prepared image and the COCO preflight..."
"$PYTHON_BIN" scripts/prepare_antiuav300_ir_yolox.py \
  --out "$PREPARED_ROOT" \
  --verify-only

COMMON_ENV=(
  env -u LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_VALUE"
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0
  AERO_ANTIUAV300_IR_PREPARED="$PREPARED_ROOT"
  AERO_ANTIUAV300_TRAINING_PROFILE="$TRAINING_PROFILE"
  AERO_YOLOX_OUTPUT="$OUTPUT_ROOT"
  AERO_YOLOX_MAX_EPOCHS="$EPOCHS"
  AERO_YOLOX_EVAL_INTERVAL="$EVAL_INTERVAL"
  AERO_YOLOX_PRINT_INTERVAL="$PRINT_INTERVAL"
  AERO_YOLOX_WORKERS=8
  AERO_YOLOX_SEED=0
  AERO_YOLOX_GRAD_ACCUM=2
  AERO_YOLOX_EFFECTIVE_BATCH=64
  AERO_YOLOX_METRICS_PATH="$RUN_DIR/metrics.json"
  AERO_YOLOX_PREDICTIONS_PATH="$RUN_DIR/predictions.json"
)

echo "Checking CUDA..."
"${COMMON_ENV[@]}" "$PYTHON_BIN" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)"

echo "Verifying the resolved training schedule..."
"${COMMON_ENV[@]}" "$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

from aero_ir.detect.yolox_antiuav300_exp import Exp

exp = Exp()
prepared_root = Path(os.environ["AERO_ANTIUAV300_IR_PREPARED"])
train_annotations = json.loads(
    (prepared_root / "annotations" / exp.train_ann).read_text(encoding="utf-8")
)
train_images = len(train_annotations["images"])
microbatches_per_epoch = math.ceil(train_images / 32)
optimizer_steps_per_epoch = math.ceil(
    microbatches_per_epoch / exp.gradient_accumulation_steps
)
scheduler = exp.get_lr_scheduler(-1.0, microbatches_per_epoch)
expected_total_steps = optimizer_steps_per_epoch * exp.max_epoch
if scheduler.total_iters != expected_total_steps:
    raise RuntimeError(
        f"scheduler has {scheduler.total_iters} steps, expected {expected_total_steps}"
    )
if exp.training_profile == "standard":
    if exp.warmup_epochs != 5 or exp.no_aug_epochs != 15:
        raise RuntimeError("standard YOLOX warmup/no-augmentation schedule changed")
    if scheduler.update_lr(1) >= scheduler.lr:
        raise RuntimeError("standard YOLOX schedule has no effective warmup")
    no_aug_start = scheduler.total_iters - scheduler.iters_per_epoch * exp.no_aug_epochs
    if scheduler.update_lr(no_aug_start) != scheduler.lr * exp.min_lr_ratio:
        raise RuntimeError("standard YOLOX schedule does not enter the minimum-LR final phase")
print(
    json.dumps(
        {
            "profile": exp.training_profile,
            "epochs": exp.max_epoch,
            "warmup_epochs": exp.warmup_epochs,
            "no_aug_epochs": exp.no_aug_epochs,
            "train_images": train_images,
            "microbatches_per_epoch": microbatches_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": scheduler.total_iters,
            "base_learning_rate": scheduler.lr,
            "first_step_learning_rate": scheduler.update_lr(1),
            "final_phase_learning_rate": scheduler.lr * exp.min_lr_ratio,
        },
        indent=2,
        sort_keys=True,
    )
)
PY

echo "Starting native-IR $TRAINING_PROFILE run: $RUN_ID ($EPOCHS epochs)"
"${COMMON_ENV[@]}" "$PYTHON_BIN" -m yolox.tools.train \
  -f "$PROJECT_ROOT/src/aero_ir/detect/yolox_antiuav300_exp.py" \
  -d 1 \
  -b 32 \
  --fp16 \
  -expn "$RUN_ID" \
  -l tensorboard

for artifact in metrics.json predictions.json last_epoch_ckpt.pth runtime.json train_log.txt; do
  if [[ ! -s "$RUN_DIR/$artifact" ]]; then
    echo "training did not complete: missing $RUN_DIR/$artifact" >&2
    exit 1
  fi
done

echo "Anti-UAV300 native-IR run complete: $RUN_DIR"
echo "This run does not clear the paired RGB-to-IR registration or generator-training HOLD."
