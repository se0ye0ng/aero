#!/usr/bin/env bash
# No reviewer input. Bounded hypothesis screen, not full training or qualification.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
DEVICE="${AERO_DEVICE:-cuda}"
OUTPUT_ROOT="${AERO_AUTO_ROI_OUTPUT:-$PROJECT_ROOT/experiments/registration_auto_roi_gpu_v1}"
EXTERNAL_ROOT="${AERO_EXTERNAL_ROOT:-$PROJECT_ROOT/experiments/external}"
BASELINE="${AERO_MATCH_BASELINE:-$PROJECT_ROOT/experiments/detector_free_train16_01/report.json}"
for value in "$PYTHON_BIN" "$DATA_ROOT" "$OUTPUT_ROOT" "$EXTERNAL_ROOT" "$BASELINE"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    printf 'Multiline path rejected: %q\n' "$value" >&2
    exit 2
  fi
done
if [[ "$DEVICE" != cpu && "$DEVICE" != cuda ]]; then
  printf 'AERO_DEVICE must be cpu or cuda.\n' >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing to overwrite: %q; choose a fresh AERO_AUTO_ROI_OUTPUT.\n' "$OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$BASELINE" || ! -f "$DATA_ROOT/label_new/train.json" ||
      ! -f "$EXTERNAL_ROOT/weights_xoftr_640.ckpt" ]]; then
  printf 'Missing Python, baseline, train data, or pinned XoFTR weights.\n' >&2
  exit 2
fi
# Preserve CUDA_VISIBLE_DEVICES supplied by the scheduler; do not select another GPU.
COMMON=(env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
"${COMMON[@]}" "$PYTHON_BIN" -B -m pytest -q tests/test_registration_auto_roi.py \
  tests/test_xoftr_hud_masks.py tests/test_xoftr_overlay.py
if [[ "$DEVICE" == cuda ]]; then
  "${COMMON[@]}" "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").item())'
fi
# Two predeclared resolutions, not selection on evaluation accuracy. Both include
# raw/inpainted inputs, identical retained support, and shuffled sequence controls.
for side in 256 512; do
  "${COMMON[@]}" "$PYTHON_BIN" -B -m scripts.probe_registration_auto_roi \
    --root "$DATA_ROOT" --external-root "$EXTERNAL_ROOT" --baseline-report "$BASELINE" \
    --output-dir "$OUTPUT_ROOT/side$side" --device "$DEVICE" --side "$side"
done
printf 'Automatic ROI diagnostics complete: %s\n' "$OUTPUT_ROOT"
printf 'Read both report.json files; completion is NOT registration qualification.\n'
