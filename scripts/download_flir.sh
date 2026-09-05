#!/usr/bin/env bash
# Teledyne FLIR ADAS Thermal v2 - access route and integrity check.
# Nothing is downloaded automatically: the dataset is distributed under a request form.
set -euo pipefail

DEST="${AERO_FLIR_ROOT:-${AERO_DATA_ROOT:-data}/FLIR_ADAS_v2}"

cat <<'MSG'
Teledyne FLIR ADAS Thermal Dataset v2
-------------------------------------
1. Request access via the Teledyne FLIR ADAS dataset form (a mirror also exists on Kaggle).
2. Extract the archive so that the following thermal directories exist:
     <dest>/images_thermal_train/
     <dest>/images_thermal_val/
     <dest>/video_thermal_test/
3. Preserve the visible/RGB data distributed with FLIR. The Phase 1 loader must create an
   immutable pair manifest and may use only pairs that pass a registration audit; do not infer
   pairs from matching numeric COCO ids because the official still-image counts differ.
4. Re-run this script to verify the thermal layout, then run `make audit-flir` to record
   annotation hashes and integrity checks under experiments/.

Licence: see the terms accompanying the download. Do not redistribute.
MSG

if [ ! -d "$DEST" ]; then
  echo "not found: $DEST - place the extracted dataset there first." >&2
  exit 1
fi

for split in images_thermal_train images_thermal_val video_thermal_test; do
  if [ ! -d "$DEST/$split" ]; then
    echo "missing split: $DEST/$split" >&2
    exit 1
  fi
  n=$(find "$DEST/$split" -type f -name "*.jpg" | wc -l | tr -d " ")
  echo "$split: $n images"
done

echo "layout ok - run 'make audit-flir' for annotation hashes and integrity checks"
