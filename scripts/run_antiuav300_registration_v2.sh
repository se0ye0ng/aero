#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
OUTPUT_ROOT="${AERO_REGISTRATION_V2_OUTPUT:-$PROJECT_ROOT/experiments}"
INITIAL_CHECKPOINT="${AERO_REGISTRATION_V2_INITIAL:-$PROJECT_ROOT/experiments/antiuav300_registration_train_e300_seed0/antiuav300_dense_matcher_e300.pth}"
CACHE_ROOT="${AERO_REGISTRATION_V2_CACHE:-$OUTPUT_ROOT/antiuav300_registration_v2_full_train_cache}"
TRAIN_DIR="${AERO_REGISTRATION_V2_TRAIN_DIR:-$OUTPUT_ROOT/antiuav300_registration_train_geometry_v2_e300_seed0}"
SCREEN_OUT="$OUTPUT_ROOT/antiuav300_dense_registration_screen_v2.json"
FULL_OUT="$OUTPUT_ROOT/antiuav300_dense_registration_full_v2.json"
FINAL_CHECKPOINT="$TRAIN_DIR/antiuav300_dense_matcher_geometry_v2_e300.pth"
CUDA_DEVICE_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
SCREEN_SAMPLES="${AERO_REGISTRATION_V2_SCREEN_SAMPLES:-8}"
BATCH_SIZE="${AERO_REGISTRATION_V2_AUDIT_BATCH:-16}"

if [[ ! -x "$PYTHON_BIN" || ! -d "$DATA_ROOT" || ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "v2 requires the project Python, Anti-UAV300 root, and frozen v1 checkpoint" >&2
  exit 2
fi
if [[ ! "$SCREEN_SAMPLES" =~ ^[1-9][0-9]*$ || ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "screen samples and audit batch must be positive integers" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$FULL_OUT" ]]; then
  "$PYTHON_BIN" -m scripts.check_antiuav300_registration_report \
    --report "$FULL_OUT" --checkpoint "$FINAL_CHECKPOINT" --root "$DATA_ROOT" --stage full
  exit 0
fi
SCREEN_EXISTS=0
if [[ -e "$SCREEN_OUT" ]]; then
  "$PYTHON_BIN" -m scripts.check_antiuav300_registration_report \
    --report "$SCREEN_OUT" --checkpoint "$FINAL_CHECKPOINT" --root "$DATA_ROOT" \
    --stage screen --screen-samples "$SCREEN_SAMPLES"
  SCREEN_EXISTS=1
fi
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
  echo "v2 qualification requires a clean, committed protocol" >&2
  git -C "$PROJECT_ROOT" status --short >&2
  exit 2
fi

COMMON_ENV=(
  env -u LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_VALUE"
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0
)
echo "Checking CUDA for frozen registration protocol v2..."
"${COMMON_ENV[@]}" "$PYTHON_BIN" -c \
  "import kornia, torch; assert kornia.__version__ == '0.6.5'; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)"

if [[ ! -f "$FINAL_CHECKPOINT" ]]; then
  TRAIN_ARGS=()
  if [[ -f "$TRAIN_DIR/latest.pth" ]]; then
    TRAIN_ARGS+=(--resume)
  fi
  echo "Training geometry-first v2 for the frozen 300 epochs..."
  "${COMMON_ENV[@]}" "$PYTHON_BIN" -m scripts.train_antiuav300_registration_v2 \
    --root "$DATA_ROOT" \
    --initial-checkpoint "$INITIAL_CHECKPOINT" \
    --cache-root "$CACHE_ROOT" \
    --output-dir "$TRAIN_DIR" \
    --epochs 300 \
    --pairs-per-sequence 16 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --seed 0 \
    --device cuda \
    "${TRAIN_ARGS[@]}"
fi

if [[ "$SCREEN_EXISTS" -eq 0 ]]; then
  echo "Running the frozen sequence-balanced v2 train/validation screen..."
  "${COMMON_ENV[@]}" "$PYTHON_BIN" scripts/audit_antiuav300_dense_registration.py \
    --root "$DATA_ROOT" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --out "$SCREEN_OUT" \
    --splits train val \
    --samples-per-sequence "$SCREEN_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --device cuda
  "$PYTHON_BIN" -m scripts.check_antiuav300_registration_report \
    --report "$SCREEN_OUT" --checkpoint "$FINAL_CHECKPOINT" --root "$DATA_ROOT" \
    --stage screen --screen-samples "$SCREEN_SAMPLES"
fi

echo "The v2 screen passed. Running the exhaustive 199,798-pair full test..."
"${COMMON_ENV[@]}" "$PYTHON_BIN" scripts/audit_antiuav300_dense_registration.py \
  --root "$DATA_ROOT" \
  --checkpoint "$FINAL_CHECKPOINT" \
  --out "$FULL_OUT" \
  --splits train val \
  --samples-per-sequence 0 \
  --batch-size "$BATCH_SIZE" \
  --device cuda
"$PYTHON_BIN" -m scripts.check_antiuav300_registration_report \
  --report "$FULL_OUT" --checkpoint "$FINAL_CHECKPOINT" --root "$DATA_ROOT" --stage full

echo "Anti-UAV300 registration protocol v2 qualified: $FULL_OUT"
