#!/usr/bin/env bash
# MS2 (Multi-Spectral Stereo) additional-source candidate - access route and layout check.
#
# MS2 is distributed by its authors under CC BY-NC-SA 3.0 and is NOT redistributed here,
# including its download URLs: the authors publish their own download scripts on the dataset
# website, and those scripts are the only supported route. Cite Shin, Park and Kweon,
# "Deep Depth Estimation From Thermal Image", CVPR 2023. The project MIT licence does not
# relicense this data; review the dataset terms before redistributing any derivative.
set -euo pipefail

DEST="${AERO_MS2_ROOT:-${AERO_DATA_ROOT:-data}/MS2}"

cat <<MSG
MS2 Multi-Spectral Stereo dataset
---------------------------------
Destination : $DEST
Website     : https://sites.google.com/view/multi-spectral-stereo-dataset
Code / docs : https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset
Licence     : CC BY-NC-SA 3.0 - non-commercial, share-alike.

1. Open the dataset website and download the authors' own shell scripts for the splits you
   need: sync_data (stereo RGB/NIR/thermal), proj_depth (projected LiDAR depth), odom (poses).
2. Run them from $DEST. They are resumable; each split is tens of GB and the full release is
   roughly 1 TB, so fetch only the sequences you intend to use.
3. Re-run this script to verify the resulting layout.
4. Pinned upstream metadata and split lists are fetched separately, without downloading any
   imagery and without executing upstream code:

       .venv/bin/python -m scripts.fetch_ms2_metadata

Nothing is downloaded automatically here.
MSG

if [ ! -d "$DEST" ]; then
    echo "not found: $DEST - place the extracted dataset there first." >&2
    exit 1
fi

missing=0
for split in sync_data proj_depth odom; do
    if [ -d "$DEST/$split" ]; then
        n=$(find "$DEST/$split" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
        echo "$split: $n sequences"
    else
        echo "missing split: $DEST/$split" >&2
        missing=1
    fi
done

if [ "$missing" = "1" ]; then
    exit 1
fi

echo "layout ok - MS2 remains an unqualified candidate source; see docs/registration_ms2_source.md"
