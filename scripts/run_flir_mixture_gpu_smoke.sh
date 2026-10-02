#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${AERO_MIXTURE_SMOKE_OUTPUT:-$PWD/experiments/flir_mixture_gpu_smoke_01}"
PYTHON_BIN="${AERO_PYTHON:-$PWD/.venv/bin/python}"
for value in "$OUT" "$PYTHON_BIN"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo 'Paths must be on a single line.' >&2
    exit 2
  fi
done
COMMON=(env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8)
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_mixture_gpu_smoke.py
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -B -m scripts.run_flir_mixture_gpu_smoke preflight --out-dir "$OUT"
"${COMMON[@]}" "$PYTHON_BIN" -B -u -m scripts.run_flir_mixture_gpu_smoke run --out-dir "$OUT"
