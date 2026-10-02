#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${AERO_PYTHON:-$PWD/.venv/bin/python}"
OUT="${AERO_ROMA_OUTPUT:-$PWD/experiments/registration_minima_roma_gpu_01}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-/lustre/winston1214/dataset/Anti-UAV300}"
for value in "$PYTHON_BIN" "$OUT" "$DATA_ROOT"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo 'Paths must be on a single line.' >&2
    exit 2
  fi
done
COMMON=(env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
        MPLCONFIGDIR=/tmp/aero-roma-mpl)
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_roma_coordinates.py tests/test_roma_dense.py
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -B -m scripts.probe_minima_roma_gpu --preflight --root "$DATA_ROOT" --out-dir "$OUT"
"${COMMON[@]}" "$PYTHON_BIN" -B -u -m scripts.probe_minima_roma_gpu --root "$DATA_ROOT" --out-dir "$OUT"
echo "RoMa diagnostic complete: $OUT/report.json"
echo 'Completion is not registration qualification or generator-training approval.'
