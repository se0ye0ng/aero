#!/usr/bin/env bash
# Fresh SAM inference after the crop-coordinate repair, then CPU similarity search.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${AERO_ANTIUAV300_ROOT:-${AERO_DATA_ROOT:-data}/Anti-UAV300}"
PREVIOUS="${AERO_SAM_PREVIOUS:-$PROJECT_ROOT/experiments/registration_sam_train16_v1/report.json}"
OUTPUT="${AERO_SAM_V2_OUTPUT:-$PROJECT_ROOT/experiments/registration_sam_train16_v2}"
WEIGHTS="${AERO_SAM_WEIGHTS:-$PROJECT_ROOT/experiments/external/sam_vit_b_01ec64.pth}"
BASELINE="${AERO_MATCH_BASELINE:-$PROJECT_ROOT/experiments/detector_free_train16_01/report.json}"
for value in "$PYTHON_BIN" "$DATA_ROOT" "$PREVIOUS" "$OUTPUT" "$WEIGHTS" "$BASELINE"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    printf 'Multiline path rejected: %q\n' "$value" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT" ]]; then
  printf 'Refusing to overwrite %q; set a fresh AERO_SAM_V2_OUTPUT.\n' "$OUTPUT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$PREVIOUS" || ! -f "$WEIGHTS" || ! -f "$BASELINE" ||
      ! -f "$DATA_ROOT/label_new/train.json" ]]; then
  printf 'Missing v1 report, pinned SAM weights, baseline panel, dataset, or Python.\n' >&2
  exit 2
fi
COMMON=(env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
# No installation, download, scheduler visibility change, or overwrite.
"${COMMON[@]}" "$PYTHON_BIN" -B -m pytest -q tests/test_registration_roi_v2.py \
  tests/test_registration_sam.py tests/test_registration_auto_roi.py
"${COMMON[@]}" "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").item())'
"${COMMON[@]}" "$PYTHON_BIN" -B -m scripts.probe_registration_sam_v2 \
  --mode extract --device cuda --root "$DATA_ROOT" --previous-report "$PREVIOUS" \
  --baseline-report "$BASELINE" --weights "$WEIGHTS" --output-dir "$OUTPUT"
printf 'Corrected-crop SAM diagnostic complete: %s/report.json\n' "$OUTPUT"
printf 'Completion is not registration qualification or generator-training approval.\n'
