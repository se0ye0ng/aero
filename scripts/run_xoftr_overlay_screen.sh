#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec env -u LD_LIBRARY_PATH MPLCONFIGDIR="${AERO_MPL_CACHE:-/tmp/aero-matplotlib}" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}" -B \
  -m scripts.probe_xoftr_overlay \
  --root "${AERO_ANTIUAV300_ROOT:-/lustre/winston1214/dataset/Anti-UAV300}" \
  --baseline-report "${AERO_OVERLAY_BASELINE:-$PROJECT_ROOT/experiments/detector_free_train16_01/report.json}" \
  --external-root "${AERO_MATCHING_EXTERNAL:-$PROJECT_ROOT/experiments/external}" \
  --output-dir "${AERO_OVERLAY_OUTPUT:-$PROJECT_ROOT/experiments/xoftr_overlay_train16_gpu_01}" \
  --device "${AERO_DEVICE:-cuda}" "$@"
