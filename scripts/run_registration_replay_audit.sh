#!/usr/bin/env bash
# Read-only model audit AFTER both replay300 arms finish; never generator GO.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MODE="${1:-preflight}"
SAMPLES=8
EXTRA=()
case "$MODE" in
  preflight) EXTRA+=(--preflight-only) ;;
  screen) ;;
  exhaustive) SAMPLES=0 ;;
  *) echo "Usage: bash $0 [preflight|screen|exhaustive]" >&2; exit 2 ;;
esac
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-/lustre/winston1214/dataset/Anti-UAV300}"
RUN_ROOT="${AERO_REPLAY_OUTPUT:-$PROJECT_ROOT/experiments/registration_replay_e300_seed0}"
BASE_ROOT="${AERO_RESIDUAL_OUTPUT:-$PROJECT_ROOT/experiments/registration_residual_pilot_e10_seed0}"
CACHE_ROOT="${AERO_REGISTRATION_CACHE:-$PROJECT_ROOT/experiments/antiuav300_registration_v2_full_train_cache}"
OUT_DIR="${AERO_REPLAY_AUDIT_OUTPUT:-$PROJECT_ROOT/experiments/registration_replay_${MODE}_audit_01}"
for value in "$PYTHON_BIN" "$DATA_ROOT" "$RUN_ROOT" "$BASE_ROOT" "$CACHE_ROOT" "$OUT_DIR"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Path contains a newline; use a single-line value." >&2; exit 2
  fi
done
COMMON=(env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
if [[ "$MODE" == preflight ]]; then COMMON+=(CUDA_VISIBLE_DEVICES=); fi
# Keep allocated device visibility for screen/exhaustive; do not force GPU0.
exec "${COMMON[@]}" "$PYTHON_BIN" -u -m scripts.audit_registration_replay \
  --root "$DATA_ROOT" --run-root "$RUN_ROOT" --base-root "$BASE_ROOT" --cache-root "$CACHE_ROOT" \
  --out-dir "$OUT_DIR" --device cuda --samples-per-sequence "$SAMPLES" "${EXTRA[@]}"
