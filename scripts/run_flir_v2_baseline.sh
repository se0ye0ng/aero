#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
RUN_ID="flir_real_only_full_seed0_v2"
SPEC_PATH="$PROJECT_ROOT/experiments/yolox_specs/$RUN_ID.json"
RUN_DIR="$PROJECT_ROOT/experiments/yolox_runs/$RUN_ID"
FLIR_ROOT="${AERO_FLIR_ROOT:-${AERO_DATA_ROOT:-data}/FLIR_ADAS_v2}"

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$FLIR_ROOT" ]]; then
  echo "FLIR dataset root is missing: $FLIR_ROOT" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "formal v2 baseline requires a clean Git checkout" >&2
  git status --short >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -e "$SPEC_PATH" ]]; then
  "$PYTHON_BIN" scripts/run_flir_yolox.py prepare \
    --mode train \
    --run-id "$RUN_ID" \
    --root "$FLIR_ROOT" \
    --epochs 300 \
    --batch-size 32 \
    --gradient-accumulation-steps 2 \
    --effective-batch-size 64 \
    --workers 8 \
    --print-interval 10 \
    --eval-interval 10 \
    --spec-out "$SPEC_PATH"
else
  echo "Using existing content-addressed run specification: $SPEC_PATH"
  "$PYTHON_BIN" - "$SPEC_PATH" <<'PY'
import sys

from aero_ir.detect.yolox_run import load_yolox_run_spec

spec = load_yolox_run_spec(sys.argv[1])
config = spec["resolved_config"]
expected = {
    "mode": "train",
    "run_id": "flir_real_only_full_seed0_v2",
    "seed": 0,
    "epochs": 300,
    "batch_size": 32,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 64,
    "workers": 8,
    "print_interval": 10,
    "eval_interval": 10,
    "fp16": True,
}
actual = {key: config[key] for key in expected}
if actual != expected:
    raise ValueError(f"existing v2 specification differs: {actual!r}")
if spec["git"].get("dirty"):
    raise ValueError("existing v2 specification records a dirty Git checkout")
print(f"verified spec {spec['spec_sha256']} from Git {spec['git']['sha']}")
PY
fi

echo "Checking CUDA before the six-hour baseline..."
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)"

echo "Starting formal FLIR v2 baseline: $RUN_ID"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/run_flir_yolox.py execute --spec "$SPEC_PATH"

MANIFEST_PATH="$RUN_DIR/run_manifest.json"
if [[ ! -s "$MANIFEST_PATH" ]]; then
  echo "baseline did not finalize: missing $MANIFEST_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/verify_run.py --run "$MANIFEST_PATH"
echo "v2 baseline complete and statically verified: $MANIFEST_PATH"
