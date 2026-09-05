#!/usr/bin/env bash
# Anti-UAV300 paired development data + Anti-UAV410 external IR evaluation.
set -euo pipefail

ROOT="${AERO_DATA_ROOT:-data}"
SOURCE="${AERO_ANTIUAV300_ROOT:-$ROOT/Anti-UAV300}"
EXTERNAL="${AERO_ANTIUAV410_ROOT:-$ROOT/Anti-UAV410}"

cat <<'MSG'
Anti-UAV300 + Anti-UAV410
-------------------------
1. Obtain Anti-UAV300 (paired RGB/IR) and Anti-UAV410 (IR-only) from the official projects.
2. Extract them as <data-root>/Anti-UAV300 and <data-root>/Anti-UAV410, or set the
   AERO_ANTIUAV300_ROOT and AERO_ANTIUAV410_ROOT environment variables.
3. Keep splits disjoint by video sequence; never expose Anti-UAV410 evaluation frames to a
   generator or curation rule.
4. Re-run this script to verify that both roots are present. Dataset-specific adapters perform
   the final sequence and annotation validation.

Licence: see the terms in the upstream repository. Do not redistribute.
MSG

[ -d "$SOURCE" ] || { echo "not found: $SOURCE" >&2; exit 1; }
echo "paired source: $SOURCE"
if [ -d "$EXTERNAL" ]; then
    echo "external IR evaluation: $EXTERNAL"
elif [ "${AERO_REQUIRE_ANTIUAV410:-0}" = "1" ]; then
    echo "not found: $EXTERNAL" >&2
    exit 1
else
    echo "external IR evaluation pending: $EXTERNAL"
fi
