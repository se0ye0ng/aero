#!/usr/bin/env bash
# No training. Freeze v6 native forward map and test its numerical reciprocal.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
EXP="$PROJECT_ROOT/experiments"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
OUTPUT="${AERO_INVERSE_SCREEN_OUTPUT:-$EXP/antiuav300_v6_inverse_screen_01}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-/lustre/winston1214/dataset/Anti-UAV300}"
V4="$EXP/antiuav300_registration_train_coordinates_v4_e300_seed0/antiuav300_dense_matcher_bidirectional_v4_e300.pth"
V6="$EXP/antiuav300_registration_v6_pilot_e10_seed0/antiuav300_registration_v6_e10.pth"
# Preserve scheduler GPU allocation; default CPU threads avoid oversubscription.
exec env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  "$PYTHON_BIN" -m scripts.compare_antiuav300_registration_pilots \
  --root "$DATA_ROOT" --output-dir "$OUTPUT" \
  --checkpoint "v4=$V4" --checkpoint "v6=$V6" --checkpoint "v6_inverse=$V6" \
  --inverse-ir-to-rgb v6_inverse \
  --samples-per-sequence 8 --batch-size 8 --log-every 5 --device cuda "$@"
