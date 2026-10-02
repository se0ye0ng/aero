#!/usr/bin/env bash
# Additional300-epoch comparison, NOT registration qualification or generator GO.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
ARM="${1:-both}"
MODE="${2:-train}"
case "$ARM" in both|uniform|failure_aware) ;; *) echo "Unknown arm: $ARM" >&2; exit 2;; esac
case "$MODE" in preflight|smoke|train|resume|verify|compare) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
BASE_ROOT="${AERO_RESIDUAL_OUTPUT:-$PROJECT_ROOT/experiments/registration_residual_pilot_e10_seed0}"
CACHE_ROOT="${AERO_REGISTRATION_CACHE:-$PROJECT_ROOT/experiments/antiuav300_registration_v2_full_train_cache}"
OUTPUT_ROOT="${AERO_REPLAY_OUTPUT:-$PROJECT_ROOT/experiments/registration_replay_e300_seed0}"
DEVICE="cuda"
if [[ "$MODE" == smoke ]]; then
  DEVICE="cpu"
  OUTPUT_ROOT="${OUTPUT_ROOT}_cpu_smoke"
fi
for value in "$PYTHON_BIN" "$BASE_ROOT" "$CACHE_ROOT" "$OUTPUT_ROOT"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Path contains a newline; use a single-line value." >&2; exit 2
  fi
done
# Do not override scheduler-assigned GPU visibility.
COMMON=(env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
if [[ "$MODE" != train && "$MODE" != resume ]]; then COMMON+=(CUDA_VISIBLE_DEVICES=); fi
"${COMMON[@]}" CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -m pytest -q tests/test_registration_replay.py
OPTIONS=(--base-root "$BASE_ROOT" --cache-root "$CACHE_ROOT" --output-root "$OUTPUT_ROOT")
if [[ "$MODE" == compare ]]; then
  "${COMMON[@]}" CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -m scripts.train_registration_replay compare "${OPTIONS[@]}"
  exit
fi
ARMS=("$ARM")
if [[ "$ARM" == both ]]; then ARMS=(uniform failure_aware); fi
for item in "${ARMS[@]}"; do
  "${COMMON[@]}" "$PYTHON_BIN" -u -m scripts.train_registration_replay "$MODE" \
    --arm "$item" "${OPTIONS[@]}" --device "$DEVICE"
done
if [[ "$ARM" == both && ( "$MODE" == train || "$MODE" == resume || "$MODE" == verify ) ]]; then
  "${COMMON[@]}" CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -m scripts.train_registration_replay compare "${OPTIONS[@]}"
fi
