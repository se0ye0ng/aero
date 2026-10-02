#!/usr/bin/env bash
# Run on the SECOND allocated GPU / node while the original v5 pilot continues.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
INITIAL="$PROJECT_ROOT/experiments/antiuav300_registration_train_coordinates_v4_e300_seed0/antiuav300_dense_matcher_bidirectional_v4_e300.pth"
CACHE="$PROJECT_ROOT/experiments/antiuav300_registration_v2_full_train_cache"
OUTPUT="${AERO_REGISTRATION_CONTROL_OUTPUT:-$PROJECT_ROOT/experiments/antiuav300_registration_v5_control_e10_seed0}"
if [[ -v CUDA_VISIBLE_DEVICES && -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "CUDA_VISIBLE_DEVICES is explicitly empty; select your second allocated GPU." >&2
  exit 2
fi
for path in "$DATA_ROOT" "$INITIAL" "$CACHE" "$OUTPUT"; do
  if [[ "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo "A path contains a newline; use a single-line value." >&2
    exit 2
  fi
done
exec env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT/src" \
  "$PYTHON_BIN" -m scripts.train_antiuav300_registration_v5_control \
  --root "$DATA_ROOT" --initial-checkpoint "$INITIAL" --cache-root "$CACHE" \
  --output-dir "$OUTPUT" --epochs 10 --batch-size 8 --seed 0 --device cuda
