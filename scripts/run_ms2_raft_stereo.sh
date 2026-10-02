#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
OUTPUT="${AERO_MS2_RAFT_OUTPUT:-$PROJECT_ROOT/experiments/ms2_raft_stereo_01}"
if [[ "$PYTHON_BIN" == *$'\n'* || "$OUTPUT" == *$'\n'* ]]; then
  echo 'Python/output paths must not contain embedded newlines.' >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is missing: $PYTHON_BIN" >&2
  exit 2
fi
COMMON=(env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1
  MPLCONFIGDIR=/tmp/aero-ms2-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  CUBLAS_WORKSPACE_CONFIG=:4096:8)
# Preserve scheduler CUDA_VISIBLE_DEVICES for inference; hide GPUs only for tests.
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q \
  tests/test_raft_stereo_adapter.py tests/test_ms2_lidar_stereo.py tests/test_ms2_raft_stereo.py
"${COMMON[@]}" "$PYTHON_BIN" -m scripts.probe_ms2_raft_stereo \
  --out-dir "$OUTPUT" --preflight-only
"${COMMON[@]}" "$PYTHON_BIN" -m scripts.probe_ms2_raft_stereo \
  --out-dir "$OUTPUT" --device cuda
"${COMMON[@]}" CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.probe_ms2_raft_stereo \
  --out-dir "$OUTPUT" --verify-only
echo "RAFT-Stereo comparison complete: $OUTPUT/report.json"
echo 'Completion is not registration qualification or generator-training approval.'
