# Reviewed central-HUD mask ablation

This is the second overlay-control stage after the executed header-crop probe.
**It has not run on real data: human-reviewed central-HUD masks have not been supplied.**
The code refuses missing/unreviewed masks; it does not infer or invent overlay regions.

## Required human inputs

Use exactly the 16 `(sequence_id, frame_index)` pairs in
`experiments/detector_free_train16_01/report.json`. Annotate overlay strokes on the
baseline resized images: RGB `(height,width)=(360,640)`, IR `(512,640)`, not on native
RGB1920 images or header-cropped images. All image-provenance hashes are in that report.

Create single-channel uint8 PNG masks: value255 means overlay to exclude, value0 means
untouched scene. No antialiasing, color masks, resizing, interpolated masks, or invented
object boundaries. An all-zero central mask is allowed if that image has no central HUD
and a human has reviewed it. The code always adds RGB72/IR104 top header rows.
Do not design masks from model predictions or GT boxes. If a reticle overlaps the UAV,
mask the reticle honestly; do not selectively preserve predicted target matches.

Freeze a versioned manifest with the following structure. Fill real hashes, identities,
all16 pairs and paths relative to the manifest directory; placeholders are not valid.

```json
{
  "schema": "aero_hud_masks_v1",
  "reviewed_by": "actual human reviewer identifier",
  "model_blind_attestation": true,
  "baseline_report_sha256": "SHA256 of the original train16 report",
  "pairs": [
    {
      "sequence_id": "20190925_101846_1_1",
      "frame_index": 531,
      "masks": {
        "visible": {
          "path": "masks/000_visible.png",
          "sha256": "SHA256 of this mask PNG",
          "shape": [360, 640],
          "image_sha256": "visible resized_rgb_sha256 from original report"
        },
        "infrared": {
          "path": "masks/000_infrared.png",
          "sha256": "SHA256 of this mask PNG",
          "shape": [512, 640],
          "image_sha256": "infrared resized_rgb_sha256 from original report"
        }
      }
    }
  ]
}
```

Reviewer identity/attestation is recorded but cannot be independently authenticated by
the script. This is separate from physical landmark annotation; masking a HUD is not
evidence that surviving matches are correct physical correspondences.

## Run after masks are supplied

```bash
env -u LD_LIBRARY_PATH .venv/bin/python -B -m scripts.probe_xoftr_hud_masks \
  --mask-manifest /path/to/reviewed_hud_masks/manifest.json \
  --output-dir experiments/xoftr_hud_masks_train16_01 \
  --device cuda
```

Use `--device cpu` for CPU execution (the default). The command preserves GPU scheduler
allocation and does not set CUDA_VISIBLE_DEVICES. Existing output directories cannot
be overwritten. Use `-B` to avoid changing the pinned vendor checkout via bytecode files.

The original raw baseline matches are reused only after hash verification. Decoded-image
provenance, source hashes, split, pinned vendor and weights are also checked.
Three conditions use exactly the same evaluation support:

1. Raw inference with all reviewed masks/header removed from output matches.
2. Input masked pixels filled with the median of unmasked grayscale pixels.
3. Input masked pixels filled with the mean of unmasked grayscale pixels.

Fill occurs at the same dimensions with no inpainting or resizing. Statistics exclude
all masked pixels. Integer fill values are rounded to nearest. For evaluation only,
the merged mask is dilated by8 pixels in Chebyshev distance (17x17 square kernel),
and matches are removed if **either endpoint** is excluded. Both inference directions
are filtered before computing reciprocity and other metrics.

Per-frame reports record masked/excluded ROI fractions, eligible target pixel centres,
and the common fully-retained target subset. Coverage is computed at pixel centres;
tiny boxes with no pixel centres return unavailable fractions rather than false100%.
Matches and report provenance are saved. Compare both filling variants: conclusions
that depend on mean versus median are sensitive to artificial fill boundaries/content.
All matching/box/Hull scores remain proxies; generator eligibility stays HOLD.
