#!/usr/bin/env bash
# Matched 10-epoch engineering pilot; not a 300-epoch final experiment or GO gate.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-/lustre/winston1214/dataset/Anti-UAV300}"
OUTPUT_ROOT="${AERO_V7_OUTPUT:-$PROJECT_ROOT/experiments/antiuav300_registration_v7_pilot_e10_seed0}"
ARM="${1:-both}"
MODE="${2:-run}"
case "$ARM" in
  both) ARMS=(geometry geometry_mind) ;;
  geometry|geometry_mind) ARMS=("$ARM") ;;
  *) echo "Usage: bash $0 [both|geometry|geometry_mind] [run|resume|preflight]" >&2; exit 2 ;;
esac
case "$MODE" in run|resume|preflight) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
for path in "$PYTHON_BIN" "$DATA_ROOT" "$OUTPUT_ROOT"; do
  if [[ "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo "Path contains a newline; use a single-line value." >&2
    exit 2
  fi
done
if [[ "$MODE" != preflight && -v CUDA_VISIBLE_DEVICES && -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty; acquire a scheduler GPU allocation." >&2
  exit 2
fi
COMMON=(env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
for name in "${ARMS[@]}"; do
  OUT="$OUTPUT_ROOT/$name"
  if [[ -e "$OUT" && "$MODE" != resume ]]; then
    echo "Refusing to overwrite $OUT; use resume or a fresh AERO_V7_OUTPUT." >&2
    exit 2
  fi
done
echo "Running CPU coordinate/shared-transform/training regression tests..."
"${COMMON[@]}" "$PYTHON_BIN" -m pytest -q tests/test_registration_shared_velocity.py tests/test_registration_v7.py
if [[ "$MODE" != preflight ]]; then
  "${COMMON[@]}" "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.ones(1,device='cuda'); print(torch.cuda.get_device_name(0),x.item())"
fi
for name in "${ARMS[@]}"; do
  EXTRA=()
  if [[ "$MODE" == resume && -e "$OUTPUT_ROOT/$name" ]]; then EXTRA+=(--resume); fi
  if [[ "$MODE" == preflight ]]; then EXTRA+=(--preflight-only); fi
  "${COMMON[@]}" "$PYTHON_BIN" -u -m scripts.train_antiuav300_registration_v7 \
    --root "$DATA_ROOT" --output-dir "$OUTPUT_ROOT/$name" --arm "$name" \
    --device cuda --epochs 10 --batch-size 8 --seed 0 "${EXTRA[@]}"
done
echo "v7 $MODE finished. Inspect both final_train_screen.json files; generator qualification remains HOLD."
