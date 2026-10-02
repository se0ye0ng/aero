#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${AERO_PYTHON:-$PWD/.venv/bin/python}"
OUT="${AERO_TILED_MATCHING_OUTPUT:-$PWD/experiments/registration_antiuav_tiled_matching_gpu_01}"
if [[ "$OUT" == *$'\n'* || "$OUT" == *$'\r'* ]]; then
  echo 'Output path must be on a single line.' >&2
  exit 2
fi
COMMON=(env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        MPLCONFIGDIR=/tmp/aero-tiled-matching-mpl)
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_tiled_matching.py tests/test_registration_rgb_resolution.py
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.probe_antiuav_tiled_matching --preflight --out-dir "$OUT"
"${COMMON[@]}" "$PYTHON_BIN" -B -u -m scripts.probe_antiuav_tiled_matching --out-dir "$OUT"
echo "Diagnostic complete: $OUT/report.json"
echo 'Completion is not registration qualification or generator training approval.'
