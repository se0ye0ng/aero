# Reviewed silhouette diagnostic: prepared, real evaluation blocked

The implementation is ready, but **no reviewed RGB/IR silhouette masks currently
exist for this experiment**. No real mask experiment has been run. Box rectangles,
automatically produced SAM masks, or one modality's warped mask are not silently
substituted for independent human-reviewed outlines.

This optional diagnostic evaluates whether mask overlap or contour distance
prefers better candidate registration. It is not independent pixel correspondence
accuracy: identical silhouettes can hide different interior correspondences.
All reports retain registration/generator HOLD.

## Required input and coordinate contract

Provide one manifest per reviewed train pair. It must identify the exact source
and target native frames, hashes of images and binary masks, reviewer identity
and date, approval that both masks describe the same physical outline, the frozen
checkpoint artifact/hash, and an explicitly saved candidate-map array/hash.
Review is an auditable human attestation, not an automatically verified fact.

Masks are `.npy` arrays of shape `(native_image_height, native_image_width)` with
0/1 values. Each mask must match its corresponding hashed image dimensions;
no resizing or coordinate inference is performed. Source and target dimensions
may differ. Do not pass 256x256 cache masks as native-camera annotations.
Reviewer/date, pair ID, and frame IDs must be actual nonempty strings; null or
numeric substitutes fail validation.

Maps are a float `.npy` array `(N, target_height, target_width, 2)`, in x/y order,
mapping each **target-output pixel to source-input coordinates**, normalized for
pixel-center `grid_sample(..., align_corners=False)`. They are sampling maps,
not displacement fields. NaNs/infinities fail. Native-grid maps must be exported
explicitly from the reviewed coordinate pipeline; the CLI does not guess frame
indices, upsample cached fields, or reconstruct missing map provenance.
Native arrays may require substantial RAM: 75 float32 maps at 640x512 take about
188 MiB before masks/warping; at 1280x720 they take about 527 MiB. Intermediate
arrays add to this. Use a suitably provisioned CPU allocation and keep the
reviewed resolution explicit rather than silently downsampling.

For the existing fine search, export 75 maps generated independently of masks
from the frozen v6 field (source-coordinate shifts -2,-1,0,1,2 in 256-grid pixel
units, scales .98,1,1.02), and preserve that unit convention in candidate specs.
The diagnostic also supports other explicit candidate arrays but makes no claim
that they are the same protocol. Include the unchanged-map baseline index.

Minimal schema below uses three candidates purely to illustrate the interface;
replace all identifiers, paths, hashes, and review fields with actual evidence.
**Do not set approval true until the human review has actually occurred.**

```json
{
  "schema_version": 1,
  "pair_id": "EXACT_TRAIN_SEQUENCE_AND_FRAME_PAIR",
  "split": "train",
  "validation_or_test_access": "none",
  "review": {
    "approved": false,
    "same_physical_outline": false,
    "reviewer_id": "ACTUAL_REVIEWER",
    "reviewed_at": "ACTUAL_REVIEW_DATE"
  },
  "source": {
    "frame_id": "EXACT_RGB_FRAME_ID",
    "image_grid": "native",
    "image": {"path": "rgb.png", "sha256": "ACTUAL_SHA256"},
    "mask": {"path": "rgb_mask.npy", "sha256": "ACTUAL_SHA256"}
  },
  "target": {
    "frame_id": "EXACT_IR_FRAME_ID",
    "image_grid": "native",
    "image": {"path": "ir.png", "sha256": "ACTUAL_SHA256"},
    "mask": {"path": "ir_mask.npy", "sha256": "ACTUAL_SHA256"}
  },
  "checkpoint": {"path": "frozen_v6.pth", "sha256": "ACTUAL_SHA256"},
  "maps": {"path": "native_sampling_maps.npy", "sha256": "ACTUAL_SHA256"},
  "map_convention": "target_to_source_normalized_pixel_centers_align_corners_false",
  "candidate_generation_uses_masks": false,
  "baseline_index": 0,
  "candidate_specs": [
    {"is_unchanged": true, "description": "frozen baseline"},
    {"is_unchanged": false, "description": "explicit candidate one"},
    {"is_unchanged": false, "description": "explicit candidate two"}
  ]
}
```

Paths resolve relative to the manifest. No downloaded models, new dependencies,
SAM installation, or GPU are required. The arrays must be exported/provided
separately; this interface deliberately does not manufacture them from boxes.

## Measurements and safeguards

- Dice loss uses bilinearly warped source masks and binary target masks.
- Symmetric boundary Chamfer averages each contour's nearest Euclidean distance
  to the other contour, giving directions equal weight. Boundaries use an
  eight-neighbor erosion; warped masks threshold at .5. Units are **target native
  pixels**, not 256-grid pixels or physical distance. Chamfer is a diagnostic,
  not a differentiable training implementation.
- One common geometric support is fixed for all candidates. Nonempty masks and
  full foreground containment within a one-pixel-eroded support are required,
  so difficult contour sections cannot simply be clipped away and rewarded.
- Empty masks and boundary-clipped foreground abstain. If any candidate is
  ineligible, neither score selects a winner from the remaining subset.
- Entirely flat scores abstain; ties favor unchanged baseline. Dice and Chamfer
  select independently and are not combined using retrospectively chosen weights.
- Human approval or artifact/hash/shape errors fail before any report is written.
  Existing output reports are never overwritten. No checkpoint is deserialized
  or trained; its hash records candidate provenance rather than model execution.

## Command after inputs are reviewed

```bash
.venv/bin/python -m scripts.probe_registration_silhouette \
  --manifest /absolute/path/to/reviewed_train_pair.json \
  --output experiments/reviewed_silhouette_pair01/report.json
```

Unit tests cover known translations, zero-mask abstention, invalid/non-finite
arrays, clipped support, review/hash/split failures, and the fact that even a
same-mask perfect score never becomes independent correspondence qualification.

**Current next dependency:** obtain reviewed paired outlines and explicitly
export matching native-coordinate candidate maps. Until then this is validated
preparatory code, not evidence that silhouette constraints solve registration.
