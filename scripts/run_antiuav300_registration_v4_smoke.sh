#!/usr/bin/env bash
# Short engineering check only. Never starts/resumes a 300-epoch training run.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
CHECKPOINT="$PROJECT_ROOT/experiments/antiuav300_registration_train_geometry_v2_e300_seed0/antiuav300_dense_matcher_geometry_v2_e300.pth"
OUTPUT="${AERO_REGISTRATION_V4_SMOKE_OUTPUT:-$PROJECT_ROOT/experiments/antiuav300_registration_v4_gpu_smoke_v1}"
for path in "$PYTHON_BIN" "$DATA_ROOT" "$OUTPUT"; do
  if [[ "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo "Path contains a newline; re-enter the corresponding AERO_* variable on one line." >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" || ! -d "$DATA_ROOT" || ! -f "$CHECKPOINT" ]]; then
  echo "Required project Python, Anti-UAV300 root, or completed v2 checkpoint is missing." >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Output exists; choose a NEW AERO_REGISTRATION_V4_SMOKE_OUTPUT. Nothing overwritten." >&2
  exit 2
fi
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "CPU preflight..."
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
  "$PYTHON_BIN" -m pytest -q tests/test_registration_coordinates_v4.py tests/test_registration_v4_smoke.py
echo "GPU smoke: batch 8, FP32, 3 optimizer updates. No full training or weight saves."
env -u LD_LIBRARY_PATH CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
  "$PYTHON_BIN" -u -m scripts.smoke_antiuav300_registration_v4 \
  --root "$DATA_ROOT" --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT" --steps 3
