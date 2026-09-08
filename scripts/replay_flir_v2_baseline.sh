#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
MANIFEST_PATH="$PROJECT_ROOT/experiments/yolox_runs/flir_real_only_full_seed0_v2/run_manifest.json"

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -s "$MANIFEST_PATH" ]]; then
  echo "v2 baseline manifest is missing: $MANIFEST_PATH" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "formal v2 replay requires a clean Git checkout" >&2
  git status --short >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Checking the frozen v2 artifacts before requesting CUDA..."
"$PYTHON_BIN" scripts/verify_run.py --run "$MANIFEST_PATH"

echo "Checking CUDA before the approximately seven-hour replay..."
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)"

echo "Replaying the frozen v2 manifest and comparing metrics..."
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/verify_run.py --run "$MANIFEST_PATH" --execute
