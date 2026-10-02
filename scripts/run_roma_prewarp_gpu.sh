#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${AERO_PYTHON:-$PWD/.venv/bin/python}"
OUT="${AERO_ROMA_PREWARP_OUTPUT:-$PWD/experiments/registration_roma_prewarp_gpu_01}"
for value in "$PYTHON_BIN" "$OUT"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo 'Paths must be on a single line.' >&2
    exit 2
  fi
done
COMMON=(env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aero-roma-prewarp)
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_registration_prewarp.py
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -B -m scripts.probe_roma_prewarp_gpu preflight --out-dir "$OUT"
"${COMMON[@]}" "$PYTHON_BIN" -B -u -m scripts.probe_roma_prewarp_gpu run --out-dir "$OUT"
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -B -m scripts.probe_roma_prewarp_gpu verify --out-dir "$OUT"
echo "Prewarp comparison complete: $OUT/report.json (not qualification approval)"
