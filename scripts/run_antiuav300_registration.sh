#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
CHECKPOINT="${AERO_SUPERFUSION_CHECKPOINT:-$PROJECT_ROOT/experiments/checkpoints/superfusion/RoadScene.pth}"
OUTPUT_ROOT="${AERO_REGISTRATION_OUTPUT:-$PROJECT_ROOT/experiments}"
TRAIN_CACHE="${AERO_REGISTRATION_CACHE:-$OUTPUT_ROOT/antiuav300_registration_train_cache_s16}"
TRAIN_DIR="${AERO_REGISTRATION_TRAIN_DIR:-$OUTPUT_ROOT/antiuav300_registration_train_e300_seed0}"
CUDA_DEVICE_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${AERO_REGISTRATION_BATCH_SIZE:-16}"
SCREEN_SAMPLES="${AERO_REGISTRATION_SCREEN_SAMPLES:-8}"
CHECKPOINT_SHA256="09337dddeb5c80b5c6b484f93298c355dd2a499b47260a9a0e03ae11fdf9a416"
CHECKPOINT_URL="https://raw.githubusercontent.com/Linfeng-Tang/SuperFusion/bee015a8938bee549d0132b80c1daf161ff80660/checkpoint/RoadScene.pth"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Anti-UAV300 root is missing: $DATA_ROOT" >&2
  exit 2
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "AERO_REGISTRATION_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "$SCREEN_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "AERO_REGISTRATION_SCREEN_SAMPLES must be a positive integer" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SCREEN_OUT="$OUTPUT_ROOT/antiuav300_dense_registration_screen.json"
FULL_OUT="$OUTPUT_ROOT/antiuav300_dense_registration_audit.json"
FINE_TUNED_CHECKPOINT="$TRAIN_DIR/antiuav300_dense_matcher_e300.pth"

if [[ -e "$FULL_OUT" ]]; then
  "$PYTHON_BIN" -m scripts.check_antiuav300_registration_report \
    --report "$FULL_OUT" --checkpoint "$FINE_TUNED_CHECKPOINT" \
    --root "$DATA_ROOT" --stage full
  exit 0
fi
SCREEN_EXISTS=0
if [[ -e "$SCREEN_OUT" ]]; then
  "$PYTHON_BIN" -m scripts.check_antiuav300_registration_report \
    --report "$SCREEN_OUT" --checkpoint "$FINE_TUNED_CHECKPOINT" \
    --root "$DATA_ROOT" --stage screen --screen-samples "$SCREEN_SAMPLES"
  SCREEN_EXISTS=1
fi
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
  echo "registration qualification requires a clean Git checkout" >&2
  git -C "$PROJECT_ROOT" status --short >&2
  exit 2
fi

mkdir -p "$(dirname -- "$CHECKPOINT")" "$OUTPUT_ROOT"
if [[ ! -f "$CHECKPOINT" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download the pinned SuperFusion checkpoint" >&2
    exit 2
  fi
  TEMP_CHECKPOINT="$(mktemp "${CHECKPOINT}.partial.XXXXXX")"
  trap 'rm -f -- "$TEMP_CHECKPOINT"' EXIT
  echo "Downloading the pinned SuperFusion RoadScene checkpoint..."
  curl --fail --location --retry 3 --output "$TEMP_CHECKPOINT" "$CHECKPOINT_URL"
  printf '%s  %s\n' "$CHECKPOINT_SHA256" "$TEMP_CHECKPOINT" | sha256sum --check --status
  mv -- "$TEMP_CHECKPOINT" "$CHECKPOINT"
  trap - EXIT
fi
printf '%s  %s\n' "$CHECKPOINT_SHA256" "$CHECKPOINT" | sha256sum --check --status || {
  echo "SuperFusion checkpoint hash mismatch: $CHECKPOINT" >&2
  exit 2
}

COMMON_ENV=(
  env -u LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_VALUE"
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0
)

echo "Checking CUDA and the pinned registration dependency..."
"${COMMON_ENV[@]}" "$PYTHON_BIN" -c \
  "import kornia, torch; assert kornia.__version__ == '0.6.5', kornia.__version__; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda, 'kornia', kornia.__version__)"

if [[ ! -f "$FINE_TUNED_CHECKPOINT" ]]; then
  TRAIN_ARGS=()
  if [[ -f "$TRAIN_DIR/latest.pth" ]]; then
    TRAIN_ARGS+=(--resume)
  fi
  echo "Fine-tuning dense registration on official train sequences only (300 epochs)..."
  "${COMMON_ENV[@]}" "$PYTHON_BIN" -m scripts.train_antiuav300_registration \
    --root "$DATA_ROOT" \
    --initial-checkpoint "$CHECKPOINT" \
    --cache-root "$TRAIN_CACHE" \
    --output-dir "$TRAIN_DIR" \
    --epochs 300 \
    --samples-per-sequence 16 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --seed 0 \
    --device cuda \
    "${TRAIN_ARGS[@]}"
fi
if [[ ! -s "$FINE_TUNED_CHECKPOINT" ]]; then
  echo "fine-tuned registration checkpoint is missing: $FINE_TUNED_CHECKPOINT" >&2
  exit 1
fi

if [[ "$SCREEN_EXISTS" -eq 0 ]]; then
  echo "Running the sequence-balanced train/validation screen..."
  "${COMMON_ENV[@]}" "$PYTHON_BIN" scripts/audit_antiuav300_dense_registration.py \
    --root "$DATA_ROOT" \
    --checkpoint "$FINE_TUNED_CHECKPOINT" \
    --out "$SCREEN_OUT" \
    --splits train val \
    --samples-per-sequence "$SCREEN_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --device cuda
  "$PYTHON_BIN" -m scripts.check_antiuav300_registration_report \
    --report "$SCREEN_OUT" --checkpoint "$FINE_TUNED_CHECKPOINT" \
    --root "$DATA_ROOT" --stage screen --screen-samples "$SCREEN_SAMPLES"
fi

echo "The screen passed. Running every usable train/validation pair..."
"${COMMON_ENV[@]}" "$PYTHON_BIN" scripts/audit_antiuav300_dense_registration.py \
  --root "$DATA_ROOT" \
  --checkpoint "$FINE_TUNED_CHECKPOINT" \
  --out "$FULL_OUT" \
  --splits train val \
  --samples-per-sequence 0 \
  --batch-size "$BATCH_SIZE" \
  --device cuda

"$PYTHON_BIN" -m scripts.check_antiuav300_registration_report \
  --report "$FULL_OUT" --checkpoint "$FINE_TUNED_CHECKPOINT" \
  --root "$DATA_ROOT" --stage full

echo "Anti-UAV300 RGB-to-IR registration qualification passed: $FULL_OUT"
