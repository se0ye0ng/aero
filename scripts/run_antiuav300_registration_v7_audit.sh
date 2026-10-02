#!/usr/bin/env bash
# Evaluation only, after the matched v7 pilots. Never starts generator training.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MODE="${1:-screen}"
SAMPLES=8
EXTRA=()
case "$MODE" in
  screen) ;;
  exhaustive) SAMPLES=0 ;;
  preflight) EXTRA+=(--preflight-only) ;;
  *) echo "Usage: bash $0 [screen|exhaustive|preflight]" >&2; exit 2 ;;
esac
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
RUN_ROOT="${AERO_V7_OUTPUT:-$PROJECT_ROOT/experiments/antiuav300_registration_v7_pilot_e10_seed0}"
OUT_DIR="${AERO_V7_AUDIT_OUTPUT:-$PROJECT_ROOT/experiments/registration_v7_${MODE}_audit_01}"
for path in "$PYTHON_BIN" "$DATA_ROOT" "$RUN_ROOT" "$OUT_DIR"; do
  if [[ "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo "Path contains a newline; use a single-line value." >&2
    exit 2
  fi
done
# Preserve scheduler CUDA_VISIBLE_DEVICES. The Python entry point verifies both
# completed pilots and the training screen before any validation pixel inference.
exec env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT/src" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON_BIN" -u -m scripts.audit_antiuav300_registration_v7 \
  --root "$DATA_ROOT" --run-root "$RUN_ROOT" --out-dir "$OUT_DIR" \
  --device cuda --samples-per-sequence "$SAMPLES" "${EXTRA[@]}"
