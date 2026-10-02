#!/usr/bin/env bash
# Bounded diagnostics, never full registration/generator training.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${AERO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
CACHE_ROOT="${AERO_REGISTRATION_CACHE:-$PROJECT_ROOT/experiments/antiuav300_registration_v2_full_train_cache}"
CHECKPOINT="${AERO_REGISTRATION_CHECKPOINT:-$PROJECT_ROOT/experiments/antiuav300_registration_v6_pilot_e10_seed0/antiuav300_registration_v6_e10.pth}"
OUTPUT_ROOT="${AERO_ALTERNATIVES_OUTPUT:-$PROJECT_ROOT/experiments/registration_alternatives_repeat01}"
for value in "$PYTHON_BIN" "$CACHE_ROOT" "$CHECKPOINT" "$OUTPUT_ROOT"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    printf 'Multiline path rejected: %q\n' "$value" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" || ! -f "$CACHE_ROOT/manifest.json" || ! -f "$CHECKPOINT" ]]; then
  printf 'Required Python, cache manifest, or checkpoint missing.\n' >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing to overwrite output: %q\n' "$OUTPUT_ROOT" >&2
  exit 2
fi
COMMON=(env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
"${COMMON[@]}" "$PYTHON_BIN" -B -m pytest -q \
  tests/test_registration_mi.py tests/test_registration_ngcc.py \
  tests/test_xoftr_overlay.py tests/test_match_geometry.py \
  tests/test_registration_physical_review.py
"${COMMON[@]}" "$PYTHON_BIN" -B -m scripts.probe_registration_mi \
  --cache-root "$CACHE_ROOT" --checkpoint "$CHECKPOINT" --sequences 160 \
  --output-dir "$OUTPUT_ROOT/mi"
"${COMMON[@]}" "$PYTHON_BIN" -B -m scripts.probe_registration_ngcc \
  --cache-root "$CACHE_ROOT" --checkpoint "$CHECKPOINT" --sequences 160 \
  --output-dir "$OUTPUT_ROOT/ngcc"
printf 'CPU diagnostics saved to %s; this does not grant qualification.\n' "$OUTPUT_ROOT"
