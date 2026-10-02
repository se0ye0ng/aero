#!/usr/bin/env bash
# New10-epoch matched pilot. Does not resume/relabel v7 or grant qualification.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
ARM="${1:-both}"
MODE="${2:-train}"
case "$ARM" in both|predictor_continuation|residual_head) ;; *) echo "Unknown arm: $ARM" >&2; exit 2;; esac
case "$MODE" in train|resume|preflight|smoke|verify) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
BASE_ROOT="${AERO_V7_OUTPUT:-$PROJECT_ROOT/experiments/antiuav300_registration_v7_pilot_e10_seed0}"
CACHE_ROOT="${AERO_REGISTRATION_CACHE:-$PROJECT_ROOT/experiments/antiuav300_registration_v2_full_train_cache}"
OUTPUT_ROOT="${AERO_RESIDUAL_OUTPUT:-$PROJECT_ROOT/experiments/registration_residual_pilot_e10_seed0}"
for value in "$PYTHON_BIN" "$BASE_ROOT" "$CACHE_ROOT" "$OUTPUT_ROOT"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Path contains a newline; use a single-line value." >&2
    exit 2
  fi
done
ARMS=("$ARM")
if [[ "$ARM" == both ]]; then ARMS=(predictor_continuation residual_head); fi
# Keep scheduler CUDA_VISIBLE_DEVICES; do not select another user's GPU.
COMMON=(env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
"${COMMON[@]}" "$PYTHON_BIN" -m pytest -q tests/test_registration_residual_velocity.py \
  tests/test_registration_residual_pilot.py
for item in "${ARMS[@]}"; do
  "${COMMON[@]}" "$PYTHON_BIN" -u -m scripts.train_registration_residual_pilot "$MODE" \
    --arm "$item" --base-root "$BASE_ROOT" --cache-root "$CACHE_ROOT" \
    --output-root "$OUTPUT_ROOT" --device cuda
done
if [[ "$ARM" == both && ( "$MODE" == train || "$MODE" == resume || "$MODE" == verify ) ]]; then
  "${COMMON[@]}" "$PYTHON_BIN" -m scripts.compare_registration_residual_pilot \
    --run-root "$OUTPUT_ROOT" --cache-root "$CACHE_ROOT" --out "$OUTPUT_ROOT/comparison.json"
fi
