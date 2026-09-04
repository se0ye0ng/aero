#!/usr/bin/env bash
# Teledyne FLIR ADAS Thermal v2 - access route and integrity check.
# Nothing is downloaded automatically: the dataset is distributed under a request form.
set -euo pipefail

DEST="${AERO_DATA_ROOT:-data}/flir_adas_v2"

cat <<'MSG'
Teledyne FLIR ADAS Thermal Dataset v2
-------------------------------------
1. Request access via the Teledyne FLIR ADAS dataset form (a mirror also exists on Kaggle).
2. Extract the archive so that the following exist:
     <dest>/images_thermal_train/
     <dest>/images_thermal_val/
     <dest>/video_thermal_test/
3. Re-run this script to verify the layout and record checksums.

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

find "$DEST" -maxdepth 2 -name "*.json" -exec shasum -a 256 {} \; > "$DEST/checksums.txt"
echo "wrote $DEST/checksums.txt"
