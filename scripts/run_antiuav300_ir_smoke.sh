#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
PREPARED_ROOT="${AERO_ANTIUAV300_IR_PREPARED:-$PROJECT_ROOT/experiments/antiuav300_ir_yolox}"
OUTPUT_ROOT="${AERO_YOLOX_OUTPUT:-$PROJECT_ROOT/experiments/yolox_runs}"
RUN_ID="${AERO_ANTIUAV300_RUN_ID:-antiuav300_native_ir_smoke_seed0}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "invalid AERO_ANTIUAV300_RUN_ID: $RUN_ID" >&2
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

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Verifying every prepared image and the COCO preflight..."
"$PYTHON_BIN" scripts/prepare_antiuav300_ir_yolox.py \
  --out "$PREPARED_ROOT" \
  --verify-only

COMMON_ENV=(
  env -u LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES=0
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0
  AERO_ANTIUAV300_IR_PREPARED="$PREPARED_ROOT"
  AERO_YOLOX_OUTPUT="$OUTPUT_ROOT"
  AERO_YOLOX_MAX_EPOCHS=1
  AERO_YOLOX_EVAL_INTERVAL=1
  AERO_YOLOX_PRINT_INTERVAL=5
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

echo "Starting native-IR engineering smoke: $RUN_ID"
"${COMMON_ENV[@]}" "$PYTHON_BIN" -m yolox.tools.train \
  -f "$PROJECT_ROOT/src/aero_ir/detect/yolox_antiuav300_exp.py" \
  -d 1 \
  -b 32 \
  --fp16 \
  -expn "$RUN_ID" \
  -l tensorboard

for artifact in metrics.json predictions.json last_epoch_ckpt.pth runtime.json train_log.txt; do
  if [[ ! -s "$RUN_DIR/$artifact" ]]; then
    echo "smoke did not complete: missing $RUN_DIR/$artifact" >&2
    exit 1
  fi
done

echo "Anti-UAV300 native-IR engineering smoke complete: $RUN_DIR"
echo "This run does not clear the paired RGB-to-IR registration or generator-training HOLD."
