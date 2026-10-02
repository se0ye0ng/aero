#!/usr/bin/env bash
# Pretrained inference only: no registration/generator training or automatic GO.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
EXTERNAL_ROOT="${AERO_MATCHING_EXTERNAL:-$PROJECT_ROOT/experiments/external}"
OUTPUT="${AERO_MATCHING_OUTPUT:-$PROJECT_ROOT/experiments/detector_free_train160_gpu_01}"
for required in "$EXTERNAL_ROOT/weights_xoftr_640.ckpt" "$EXTERNAL_ROOT/loftr_outdoor.ckpt" \
  "$EXTERNAL_ROOT/XoFTR/src/config/default.py" "$DATA_ROOT/label_new/train.json"; do
  if [[ ! -f "$required" ]]; then
    printf 'Required input missing: %q\nSee docs/detector_free_matching.md for setup.\n' "$required" >&2
    exit 1
  fi
done
# Preserve the scheduler's CUDA_VISIBLE_DEVICES allocation.
exec env -u LD_LIBRARY_PATH \
  MPLCONFIGDIR="${AERO_MPL_CACHE:-/tmp/aero-matplotlib}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  "$PYTHON_BIN" -B -m scripts.probe_detector_free_matching \
  --root "$DATA_ROOT" --external-root "$EXTERNAL_ROOT" --output-dir "$OUTPUT" \
  --sequences "${AERO_MATCHING_SEQUENCES:-160}" --device cuda "$@"
