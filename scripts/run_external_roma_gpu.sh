#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${AERO_PYTHON:-$PWD/.venv/bin/python}"
OUT="${AERO_EXTERNAL_ROMA_OUTPUT:-$PWD/experiments/registration_external_roma_gpu_01}"
for value in "$PYTHON_BIN" "$OUT"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo 'Paths must be on a single line.' >&2
    exit 2
  fi
done
COMMON=(env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aero-external-roma)
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_external_roma.py tests/test_roma_dense.py
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -B -m scripts.probe_external_roma preflight --out-dir "$OUT"
"${COMMON[@]}" "$PYTHON_BIN" -B -u -m scripts.probe_external_roma run --out-dir "$OUT"
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -B -m scripts.probe_external_roma verify --out-dir "$OUT"
echo "External RoMa diagnostic complete: $OUT/report.json"
echo 'Completion is not registration qualification or generator-training approval.'
