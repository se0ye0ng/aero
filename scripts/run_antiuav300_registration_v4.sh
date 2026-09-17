#!/usr/bin/env bash
# Default is corrected-coordinate audit of the completed v2 checkpoint, NOT training.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-/lustre/winston1214/dataset/Anti-UAV300}"
INITIAL="$PROJECT_ROOT/experiments/antiuav300_registration_train_geometry_v2_e300_seed0/antiuav300_dense_matcher_geometry_v2_e300.pth"
CHECKPOINT="${AERO_REGISTRATION_V4_CHECKPOINT:-$INITIAL}"
TRAIN_DIR="${AERO_REGISTRATION_V4_TRAIN_DIR:-$PROJECT_ROOT/experiments/antiuav300_registration_train_coordinates_v4_e300_seed0}"
CACHE_ROOT="${AERO_REGISTRATION_V4_CACHE:-$PROJECT_ROOT/experiments/antiuav300_registration_v2_full_train_cache}"
REPORT="${AERO_REGISTRATION_V4_REPORT:-$PROJECT_ROOT/experiments/antiuav300_coordinates_v4_screen.json}"
MODE="${1:-audit}"
if [[ "$MODE" != audit && "$MODE" != full && "$MODE" != train ]]; then
  echo "usage: bash scripts/run_antiuav300_registration_v4.sh [audit|full|train]" >&2
  exit 2
fi
for path in "$DATA_ROOT" "$CHECKPOINT" "$TRAIN_DIR" "$CACHE_ROOT" "$REPORT"; do
  if [[ "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo "A path contains a newline; unset/re-enter the AERO_* variable on one line." >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" || ! -d "$DATA_ROOT" || ! -f "$CHECKPOINT" ]]; then
  echo "required Python, dataset, or completed checkpoint is missing" >&2
  exit 2
fi
if [[ -e "$REPORT" ]]; then
  echo "Report already exists; choose a NEW AERO_REGISTRATION_V4_REPORT (no overwrite)." >&2
  exit 2
fi
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "Running CPU coordinate/gate regression tests before any GPU work..."
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
  "$PYTHON_BIN" -m pytest -q tests/test_registration_coordinates_v4.py
COMMON=(env -u LD_LIBRARY_PATH CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0)
"${COMMON[@]}" "$PYTHON_BIN" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
if [[ "$MODE" == train ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit the reviewed v4 source before a full 300-epoch run; do not mix with v3." >&2
    exit 2
  fi
  RESUME=()
  if [[ -f "$TRAIN_DIR/latest.pth" ]]; then RESUME=(--resume); fi
  "${COMMON[@]}" "$PYTHON_BIN" -m scripts.train_antiuav300_registration_v4 \
    --root "$DATA_ROOT" --initial-checkpoint "$INITIAL" \
    --cache-root "$CACHE_ROOT" --output-dir "$TRAIN_DIR" --device cuda "${RESUME[@]}"
  CHECKPOINT="$TRAIN_DIR/antiuav300_dense_matcher_bidirectional_v4_e300.pth"
fi
SAMPLES=8
if [[ "$MODE" == full ]]; then SAMPLES=0; fi
status=0
"${COMMON[@]}" "$PYTHON_BIN" -m scripts.audit_antiuav300_registration_v4 \
  --root "$DATA_ROOT" --checkpoint "$CHECKPOINT" --out "$REPORT" \
  --samples-per-sequence "$SAMPLES" --device cuda || status=$?
if [[ "$status" == 3 ]]; then
  echo "Audit saved. Exit 3 means research HOLD, not a CUDA error. Independent correspondence evidence is still required."
fi
exit "$status"
