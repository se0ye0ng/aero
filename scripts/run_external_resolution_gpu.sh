#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${AERO_PYTHON:-$PWD/.venv/bin/python}"
OUT="${AERO_EXTERNAL_RESOLUTION_OUT:-$PWD/experiments/registration_external_resolution_gpu_01}"
if [[ "$OUT" == *$'\n'* || "$OUT" == *$'\r'* ]]; then
  echo 'Output path contains a newline; supply a single-line path.' >&2
  exit 2
fi
COMMON=(env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        MPLCONFIGDIR=/tmp/aero-external-resolution-mpl)
"${COMMON[@]}" "$PYTHON_BIN" -m scripts.probe_external_resolution preflight --out-dir "$OUT"
"${COMMON[@]}" "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
"${COMMON[@]}" "$PYTHON_BIN" -m scripts.probe_external_resolution run --out-dir "$OUT"
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.probe_external_resolution verify --out-dir "$OUT"
echo "Completed and verified: $OUT/report.json"
echo 'Completion is not registration qualification, Anti-UAV approval or generator authorization.'
