#!/usr/bin/env bash
# Anti-UAV410 - small, low-contrast thermal targets. Access route and layout check.
set -euo pipefail

DEST="${AERO_DATA_ROOT:-data}/antiuav410"

cat <<'MSG'
Anti-UAV410
-----------
1. Obtain the dataset from the official Anti-UAV project repository.
2. Extract so that <dest>/{train,val,test}/ exist.
3. Re-run this script to verify the layout.

Licence: see the terms in the upstream repository. Do not redistribute.
MSG

[ -d "$DEST" ] || { echo "not found: $DEST" >&2; exit 1; }
for split in train val test; do
  [ -d "$DEST/$split" ] || { echo "missing split: $DEST/$split" >&2; exit 1; }
  echo "$split: $(find "$DEST/$split" -maxdepth 1 -type d | wc -l | tr -d " ") sequences"
done
