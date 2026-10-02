#!/usr/bin/env bash
# Human-free SAM pseudo-mask / invertible similarity screen. No full training.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
OUTPUT_ROOT="${AERO_SAM_OUTPUT:-$PROJECT_ROOT/experiments/registration_sam_train16_v1}"
WEIGHTS="${AERO_SAM_WEIGHTS:-$PROJECT_ROOT/experiments/external/sam_vit_b_01ec64.pth}"
BASELINE="${AERO_MATCH_BASELINE:-$PROJECT_ROOT/experiments/detector_free_train16_01/report.json}"
DEVICE="${AERO_DEVICE:-cuda}"
for value in "$PYTHON_BIN" "$DATA_ROOT" "$OUTPUT_ROOT" "$WEIGHTS" "$BASELINE"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    printf 'Multiline path rejected: %q\n' "$value" >&2
    exit 2
  fi
done
if [[ "$DEVICE" != cuda && "$DEVICE" != cpu ]]; then
  printf 'AERO_DEVICE must be cuda or cpu.\n' >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing to overwrite %q; choose a fresh AERO_SAM_OUTPUT.\n' "$OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$BASELINE" || ! -f "$DATA_ROOT/label_new/train.json" ]]; then
  printf 'Missing Python, fixed baseline panel, or train data.\n' >&2
  exit 2
fi
COMMON=(env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
# Never override the scheduler's CUDA_VISIBLE_DEVICES.
if [[ "$DEVICE" == cuda ]]; then
  "${COMMON[@]}" "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").item())'
fi
"$PYTHON_BIN" -m pip install --no-deps segment-anything==1.0
"${COMMON[@]}" "$PYTHON_BIN" -B -m pytest -q tests/test_registration_sam.py \
  tests/test_registration_auto_roi.py tests/test_registration_timing.py
if [[ ! -f "$WEIGHTS" ]]; then
  printf 'Downloading official SAM ViT-B checkpoint (~375 MB).\n'
  mkdir -p -- "$(dirname -- "$WEIGHTS")"
  DOWNLOAD_TMP="$(mktemp "${WEIGHTS}.download.XXXXXX")"
  # On network failure leave the uniquely named partial file for inspection, never
  # treat it as a finished checkpoint. Existing weights are not overwritten.
  curl --fail --location --retry 3 \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    --output "$DOWNLOAD_TMP"
  if [[ -e "$WEIGHTS" ]]; then
    printf 'Checkpoint appeared during download; retained temporary file %q.\n' "$DOWNLOAD_TMP" >&2
    exit 2
  fi
  mv -n -- "$DOWNLOAD_TMP" "$WEIGHTS"
fi
"${COMMON[@]}" "$PYTHON_BIN" -B -m scripts.probe_registration_sam \
  --root "$DATA_ROOT" --baseline-report "$BASELINE" --weights "$WEIGHTS" \
  --output-dir "$OUTPUT_ROOT" --device "$DEVICE"
printf 'SAM automatic contour diagnostic completed: %s/report.json\n' "$OUTPUT_ROOT"
printf 'This is NOT 300-epoch training or registration qualification. No human review is required to execute it.\n'
