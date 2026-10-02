# Frozen DINOv2 whole-image correspondence screen

## Scope and model provenance

This is a bounded **train-only** pretrained patch-feature diagnostic, not DINO
fine-tuning or a registration qualification. Ground-truth boxes enter evaluation
and visual review only; they never crop model inputs, select tokens, or choose
matches. No validation/test access or generator eligibility change occurs.

- Official [DINOv2 implementation](https://github.com/facebookresearch/dinov2),
  clean revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- ViT-S/14 with four registers, last-layer `x_norm_patchtokens` (384 features per
  patch). Class and register tokens are not matched.
- Official checkpoint URL:
  `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth`.
- SHA256 `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`.
- Direct official backbone import, `pretrained=False`, explicit local
  `torch.load(weights_only=True)` and strict state-dictionary loading. No hub
  download, random-weight fallback, or train-mode updates.

These generic visual features are not an RGB/thermal correspondence model.
Successful same-modality controls do not establish thermal pixel accuracy.

## Protocol

Reuse the exact 16 native-decoded train pairs from the earlier detector-free
screen. The baseline report hash is pinned to
`1267cfe38eb0676b779704275f3e3939debb664b74bc7fe14598e7e6e292a6d8`;
sequence/frame IDs, annotation hashes, native-decoded hashes and resized-image
hashes must match before inference. RGB starts at 640x360, IR at 640x512.

Run both whole-image conditions:

1. **Original:** retain the whole frame, resize downward to multiples of 14:
   RGB 630x350, IR 630x504.
2. **Header crop:** remove the upper 72 RGB rows and 104 IR rows before any
   feature extraction, then resize to RGB 630x280, IR 630x406. This removes
   screen text in the broad band but can remove scene content and does not
   remove central reticles or certify complete overlay removal.

RGB remains color; IR becomes grayscale repeated across three channels. Inputs
are normalized by ImageNet mean/std. Each patch contributes one centre
descriptor. Cosine similarity and mutual nearest-neighbour selection are used
without a confidence threshold tuned on these frames. Zero descriptors abstain;
ties resolve deterministically through argmax.

Points are mapped back to their own native-derived 640-wide frames using the
half-pixel resize convention and crop offsets. A 14px patch centre is at 6.5,
not 7, in pixel-centre coordinates. These are **not native sensor pixels**.
No descriptor interpolation or upsampling is represented as subpixel matching.

Report total matches, both-box routing, hull area fractions, and the number of
patch centres inside each GT target box before match filtering. Fewer than four
centres in either target box triggers a descriptive **low-target-resolution**
warning. Counts/hulls are engineering proxies, not pixel correspondence truth
or new qualification thresholds. MNN matches are reciprocal by construction;
this is not an independent reverse-model reproducibility test.

For the first and last RGB frame in both conditions, shift the already
patch-aligned model input by (+28,-28)px and compare features. Errors are
reported only where both matched endpoints are at least 42px from image edges,
in **model-input pixels**. This tests a known two-patch displacement under
same-modality appearance. Report median error and fractions within 3/7/14px;
do not silently reinterpret patch quantization as 1px registration accuracy.

For an unrelated-pair negative control, pair first-sequence RGB with
last-sequence IR in each input condition. Nonzero cosine matches are expected
without calibrated rejection and are not proof that either image pair has
physical correspondences. Save these matches as well as all real matches.

This bounded screen intentionally does **not** use GT-box crops, predicted-ROI
crops, intermediate-layer searches, feature-layer tuning, confidence searches,
or another learned registration model. A targeted ROI approach would be a
separate protocol requiring a label-free deployed ROI source.

## Verification and reproduction

Six unit tests cover half-pixel coordinate restoration, exact header cropping,
IR channel replication, RGB color retention, patch counts, MNN permutation
recovery/reverse symmetry, empty/zero/nonfinite features, and GT isolation from
match selection. Official strict weight loading and a tiny forward pass were
also checked. CPU runs use one PyTorch thread and one OpenCV thread. The
optional xFormers package is absent; official fallback execution is used.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_dinov2_matching.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m scripts.probe_dinov2_matching \
  --output-dir experiments/dinov2_matching_train16_repeat
```

Choose a fresh output directory. Report and NPZ files preserve input, weight,
source, match-array and baseline-report hashes. Visual cards show all-input
views plus GT-centred review windows; those review windows are not model inputs.

## Completed train16 results

Artifact: `experiments/dinov2_matching_train16_01/report.json`.
The CPU screen completed in **83.79s** after loading, including visual cards.
All recorded source and real-match hashes verify.

| Proxy | Original whole image | Header-cropped whole image |
|---|---:|---:|
| Evaluated pairs | 16 | 16 |
| Median MNN matches | 56.5 | 43 |
| Pairs with any both-box match | 12/16 | 12/16 |
| Total both-box matches | 17 | 17 |
| Pairs with at least four both-box matches | 1/16 | 0/16 |
| Also at least 10% hull area in both boxes | 1/16 | 0/16 |
| Fewer than four available target patch centres in either box | 9/16 | 8/16 |

All four same-modality two-patch shift controls have zero median error. Fractions
within 3 model-input pixels range **82.28%–96.95%**, not 100%; within 14 pixels
they range 89.79%–98.98%. Since centre displacements are patch-quantized, the
3px and 7px fractions coincide. The zero median is not a subpixel-accuracy claim.

Unrelated-pair controls produce **36 original / 32 header-cropped** MNN matches
and zero both-box matches. Semantic similarity and mutuality alone therefore
do not certify physical correspondences.

Visual inspection of `010_header_crop_review.png` confirms two coarse centres
on the target, not a densely covered target contour. Whole-image matches also
concentrate along image/crop boundaries; header removal does not eliminate all
non-scene correspondence artifacts. The card displays the original image for
coordinate context, but features in the header-crop condition did not see the
excluded band. All green lines are MNN by construction, not independent GT.

**Decision:** this frozen whole-image last-layer feature configuration can
provide sparse target-location hints, but does not supply the dense, distributed
physical anchors needed to qualify the current registration. It does not justify
a full registration/generator training run. Higher any-box counts than another
matcher do not establish higher pixel precision, especially at the coarse 14px
patch resolution. This result does not reject every DINO layer, resolution,
ROI-assisted method, or cross-modal fine-tuning strategy; those were not tested.

No new GPU command is necessary for this completed diagnostic. Qualification
and generator eligibility remain **HOLD**.
