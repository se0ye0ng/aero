#!/usr/bin/env bash
# One allocated GPU; evaluate all five checkpoints on each decoded frame batch.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
OUTPUT="${AERO_REGISTRATION_COMPARE_OUTPUT:-$PROJECT_ROOT/experiments/antiuav300_registration_comparison_v4_v5_v6_01}"
EXP="$PROJECT_ROOT/experiments"
for path in "$DATA_ROOT" "$OUTPUT"; do
  if [[ "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo 'A path contains a newline. Use a single-line value.' >&2
    exit 2
  fi
done
# Preserve CUDA_VISIBLE_DEVICES assigned by the cluster scheduler.
exec env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  "$PYTHON_BIN" -m scripts.compare_antiuav300_registration_pilots \
  --root "$DATA_ROOT" --output-dir "$OUTPUT" \
  --checkpoint "v4=$EXP/antiuav300_registration_train_coordinates_v4_e300_seed0/antiuav300_dense_matcher_bidirectional_v4_e300.pth" \
  --checkpoint "control=$EXP/antiuav300_registration_v5_control_e10_seed0/antiuav300_registration_v5_control_e10.pth" \
  --checkpoint "v5_seed0=$EXP/antiuav300_registration_v5_pilot_e10_seed0/antiuav300_registration_v5_e10.pth" \
  --checkpoint "v5_seed1=$EXP/antiuav300_registration_v5_pilot_e10_seed1/antiuav300_registration_v5_e10.pth" \
  --checkpoint "v6=$EXP/antiuav300_registration_v6_pilot_e10_seed0/antiuav300_registration_v6_e10.pth" \
  --samples-per-sequence 8 --batch-size 8 --log-every 10 --device cuda "$@"
