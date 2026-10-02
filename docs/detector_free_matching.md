# Pretrained XoFTR / LoFTR correspondence-quality screen

This screen tests pretrained matchers, not a new registration training run.
Pixel-level physical correspondence accuracy is **not** known from box routing,
reverse-match agreement, or homography residuals alone. Generator eligibility
remains **HOLD**.

## Models and provenance

- [XoFTR official implementation](https://github.com/OnderT/XoFTR), revision
  `e0fbea431b30be9742effbf5577c90aa8eb938f9`, clean checkout under the git-ignored
  `experiments/external/XoFTR` directory. Uses its official 640 checkpoint.
  [Paper](https://arxiv.org/abs/2404.09692).
- [LoFTR](https://github.com/zju3dv/LoFTR), Kornia 0.6.5 implementation and outdoor
  checkpoint from the distribution URL referenced in Kornia's implementation.
  No automatic downloads or random-weight fallback occur during evaluation.
- XoFTR defaults: coarse threshold 0.3, fine threshold 0.1, denser=False.
  LoFTR defaults: coarse threshold 0.2. XoFTR fine confidence and LoFTR coarse
  confidence are different scores; raw confidences are not calibrated against
  each other, and thresholds are not tuned on the evaluated frames.
- XoFTR official checkpoint SHA256:
  `6e7f24e553f76746ef49061bf44c7d11252e56057470d1421bcd4e6506af3a67`.
- LoFTR outdoor checkpoint SHA256:
  `21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec`.

The evaluation script pins and verifies weight hashes and the clean XoFTR commit.
Loaded weights use `weights_only=True` and strict state-dictionary matching.
Existing registration checkpoints, optimizer states and qualification thresholds
are not changed.

### Empty-fine-match guard

Upstream XoFTR `FineSubMatching.get_fine_sub_match` sets a single mask entry and
its confidence to 1 when all fine scores fail the threshold. Our inference
adapter intercepts that branch, returns empty matches, and records
`suppressed_fake_fine_match`. All normal valid-match branches are unchanged.
The vendor checkout and pretrained tensors themselves are unmodified.

## Protocol

1. Select evenly spaced sequence IDs from the official **train** split, one
   midpoint usable paired frame per sequence. No val/test access or fitting.
   The 16-sequence set uses the same selection rule as the earlier MIND probe.
2. Decode the original RGB 1920x1080 and IR 640x512 video frames. Resize directly
   to long side 640, keeping aspect ratio and rounding dimensions down to a
   multiple of eight: RGB 640x360, IR 640x512. Do not upscale the old 256px cache.
3. Run each matcher in both input orders. GT boxes are not provided to the model,
   do not select matches, and are used only for evaluation and visualization.
4. Save every valid finite in-bounds match and its score. Count matches in either
   and both target boxes. These are **box-routing proxies**, not match precision.
5. Check whether the reverse run returns both endpoints within 2 resized pixels.
   Repeated wrong matches can still satisfy this consistency criterion.
6. Fit a homography with RANSAC threshold 3px on a deterministic random half of
   matches and check the other half. Held-out model predictions are **not GT**;
   a low residual does not establish physical scene correctness or handle every
   depth/parallax configuration.
7. For the first and last selected sequence, run RGB against a synthetically
   shifted RGB image (+8,-8px). This supplies a known-geometry implementation
   control, not a substitute for cross-modal evaluation.
8. Pair the first RGB frame with the last sequence's IR frame as an unpaired
   negative control. Such matches must not be accepted as evidence for that pair
   having valid synchronized physical correspondences.

All distance thresholds above are in the resized image coordinates, not native
sensor pixels. Pixel correspondences from the two image resolutions are stored
in their own respective coordinate systems. RGB/IR resize shapes, source
annotation hashes, decoded-frame hashes, source hashes, weights and match-array
hashes are recorded in `report.json`.

### Screen overlays are a real confounder

The initial two-sequence review showed many high-confidence, repeatable matches
on timestamp/text overlays. A screen-space match is not a scene-point match.
The expanded diagnostic therefore also reports matches excluding any pair whose
endpoint lies in the upper 20% of either image, and repeats the homography check
on those matches. This is an **exploratory post-hoc header screen**, motivated by
the initial visual inspection, not an independently calibrated HUD segmentation.

The screen may exclude actual scene points and does not remove central reticles.
It filters output matches only: the model's attention still saw the original
overlay pixels. Thus even the non-header statistics are not certified HUD-free
physical accuracy. A later masked-input experiment would be a separate test.

### Spatial support audit

`scripts/analyze_detector_free_screen.py` verifies saved match and annotation
hashes and measures:

- reciprocal both-box pairs deduplicated by rounding all four endpoint
  coordinates to 1px;
- convex-hull area in each target box, to distinguish several clustered points
  from distributed spatial support;
- target-box transfer IoU from the non-header homography (another annotation
  proxy, not independent point accuracy).

Four pairs and a 10% hull-area fraction are descriptive exploratory counts, **not
new qualification thresholds**. Homography poles crossing the source box are
rejected. An annotation-assisted ROI count must not be described as a label-free
matching algorithm.

## Initial engineering check

Artifact: `experiments/detector_free_train2_01/report.json`.
The two-sequence CPU run took 89.4 seconds after model loading. Four known-shift
controls each produced approximately 3,000 matches, with 100% within 3px of the
known displacement. Median errors were 0.056--0.098px for XoFTR and
0.103--0.148px for LoFTR. These are same-modality synthetic-control results only.
The two real pairs yielded 17 / 3 both-box XoFTR matches, versus 0 / 0 LoFTR
matches in RGB->IR order. Two selected frames are not sufficient to qualify a
method; the 16-sequence screen expands this comparison.

## Completed 16-sequence results

Artifacts: `experiments/detector_free_train16_01/report.json` and
`spatial_audit.json`, plus all match arrays and 32 visual review cards.
CPU evaluation took **469.6 seconds** after model loading. This covers one frame
from each of 16 train sequences, not all frames or a held-out validation result.
The header analysis was introduced after the initial two-frame visual review.

The following table uses RGB->IR input order; both directions are retained in
the JSON report.

| Diagnostic | XoFTR | LoFTR outdoor |
|---|---:|---:|
| Pairs evaluated | 16 | 16 |
| Median total matches | 265.5 | 78.5 |
| Median matches outside header screen | 41 | 34 |
| Pairs with any match inside both target boxes | 11/16 | 1/16 |
| Median both-box match count | 3.5 | 0 |
| Pairs with >=4 unique reciprocal both-box pairs | 8/16 | 0/16 |
| Also >=10% hull area in both boxes | 7/16 | 0/16 |
| Both-box / source-box match counts (not precision) | 181/210 | 1/10 |
| Non-header homography box IoU, mean over available | 0.3925 (13/16) | 0.1021 (16/16) |
| Non-header homography box IoU >=0.6, count / all pairs | 5/16 | 0/16 |

The reverse-order any-both-box counts are 11/16 for XoFTR and 3/16 for LoFTR.
XoFTR's header-screened homography fit succeeds on 14 pairs, but one source-box
projection crosses a homography pole; the box metric therefore has only 13
available cases. Missing/invalid cases must not be treated as successful.

### Overlay confounding and negative control

Across all forward matches, 3,835/7,317 XoFTR matches and 591/1,845 LoFTR matches
have at least one endpoint in the top 20% band. The per-frame median fractions
are 83.2% and 51.3%, respectively. Visual inspection confirms many high-score
points are on screen text/timestamps, but **not every point in the band is
necessarily an overlay**. The central reticles remain unmasked.

Even the unpaired first-RGB/last-IR negative control returns 170 XoFTR and 72
LoFTR matches. Of these, 138 and 37 are in the header screen. After that screen,
the fitted homographies predict **zero held-out matches within 3px** for both
models. This illustrates why raw match count or confidence alone cannot certify
physical scene correspondence.

Known-translation controls again have 100% of returned matches within 3px, with
the same 0.056--0.148px median-error range as the initial check. Both pretrained
implementations are functioning; these synthetic controls are not cross-modal
accuracy scores. The XoFTR empty-fine-match fallback was suppressed by design but
was not actually triggered on this 16-pair run.

### Visual inspection, including failures

The inspected cards include the first pair, indices 2 and 3, and index 10.
These are illustrative case inspections, not a manually labeled accuracy set.

- `010_xoftr_review.png` / `010_loftr_review.png`, sequence
  `20190926_141816_1_3`, frame 500: XoFTR proposes matches distributed over the UAV
  silhouette/body, including visually plausible correspondences; LoFTR produces
  no both-box pair. This is positive candidate evidence, not subpixel GT.
- `002_xoftr_review.png`, sequence `20190925_133630_1_5`, frame 500: 1,378 total
  matches but zero both-box pairs. Buildings/overlays dominate; nearby building
  matches must not be mistaken for matching the UAV.
- The full-view top-confidence displays repeatedly highlight shared screen
  text. Green lines in these cards mean reverse-run repeatability, **not** known
  correctness. Target crops use the same point IDs in both modalities.

### Decision

**Keep XoFTR as a correspondence-candidate source; do not use LoFTR outdoor as
the primary UAV RGB/IR anchor on the strength of this screen.** XoFTR's target
coverage is substantially better here, but it is still sparse/inconsistent across
frames and all-pixel correctness is unverified. No generator-training GO follows.

The next informative comparison is XoFTR with independently defined HUD exclusion
at the input/matching stage, followed by checked target correspondences and an
expanded sequence screen. A post-hoc header filter alone has not removed the
network's exposure to overlays. Do not replace the current registration model
or launch a 300-epoch retraining run solely on these count/consistency proxies.
The 160-sequence GPU wrapper below expands the **same original-input diagnostic**;
it is not yet a HUD-masked-input experiment or a qualification pass.

## Setup and reproduction

The official sources and weights were downloaded for this task into
`experiments/external/`; they are not committed to git. The existing torch/CUDA
installation is retained. Inference-only additions are listed in
`requirements/detector-free.txt`.

On another checkout, install those requirements, clone the official XoFTR
repository at the pinned revision, and obtain:

- `experiments/external/weights_xoftr_640.ckpt` from the
  [official weights folder](https://drive.google.com/drive/folders/1RAI243OHuyZ4Weo1NiTy280bCE_82s4q),
  file ID `1oRkEGsLpPIxlulc6a7c2q5H1XTNbfkVj`;
- `experiments/external/loftr_outdoor.ckpt` from
  [Kornia's referenced LoFTR distribution](https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt).

Use a fresh output directory for every run. `-B` avoids creating untracked Python
bytecode inside the pinned external checkout.

```bash
MPLCONFIGDIR=/tmp/aero-matplotlib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -B -m scripts.probe_detector_free_matching \
  --sequences 16 --device cpu \
  --output-dir experiments/detector_free_train16_repeat

.venv/bin/python -B -m scripts.analyze_detector_free_screen \
  --report experiments/detector_free_train16_repeat/report.json
```

For optional expanded **pretrained inference**, not a training job, on an
allocated GPU:

```bash
bash scripts/run_antiuav300_detector_free_screen.sh
```

The GPU wrapper defaults to 160 train sequences, preserves the scheduler's GPU
allocation, and writes `experiments/detector_free_train160_gpu_01/`. Override
`AERO_MATCHING_OUTPUT` for repeat runs. Exit zero means diagnostic completion,
never a dense-registration or generator-training GO.

## Dense fine window comparison

The fixed input-header-cropped train16 panel was rerun on CPU with XoFTR's
`fine_matching.denser=True`. The upstream default retains the highest-confidence
fine match per window; this option retains fine-window mutual nearest matches
above the same confidence threshold. We did not change the checkpoint, input
resolution, empty-fine-match guard, local TPS policy or IoU threshold.

| Measurement | Default fine matching | Dense fine matching |
|---|---:|---:|
| Matches summed over both directions and 16 pairs | 7,224 | 62,091 |
| Directional TPS fits | 11/32 | 16/32 |
| Joint box-corner proxy passes | 1/16 | 1/16 |

The first pair's default bidirectional inference reproduced the saved baseline
within the existing 1e-4 coordinate/confidence tolerance. A same-modality known
shift produced 58,011 matches, all within three pixels of the known displacement.
An unrelated pair still produced 25 matches. These controls check implementation
and demonstrate why match count alone is not accuracy; they do not certify
cross-modal pixel correspondence. The dense setting is not promoted as a repair.

Report: `experiments/registration_antiuav_dense_fine_01/report.json`, SHA256
`642589b25b8dce6276ff2b5c7b39328819d85489babe1e3d5685665fdc383bdf`.
All 149 recorded input/source identities and saved match artifacts were checked.
Anti-UAV qualification and generator authorization remain HOLD.

## Tiled matching and spatial support

On the fixed header-cropped train16 panel, nine overlapping half-frame tiles per
image were resized to long side 640 and all 81 tile combinations were matched in
each direction. No ground-truth box selected a tile or correspondence. The
whole-image baseline was rerun on the same RTX 3090. Both variants used unchanged
confidence thresholds and the original local-consensus TPS evaluator.

| Measurement | Whole image | Tiled |
|---|---:|---:|
| Matches over both directions | 7,221 | 144,941 |
| Directional TPS fits | 11/32 | 17/32 |
| Joint box-corner IoU at least 0.6 | 1/16 | 1/16 |
| Joint passes after reciprocal endpoint filtering | 1/16 | 1/16 |
| Joint passes with all-consensus piecewise affine interpolation | 1/16 | 1/16 |

The GPU report and saved matches were verified by CPU reconstruction of all 16
pairs, not by repeating neural inference. Tiled matching's known-shift control
had 60,830 matches with PCK3 of 79.86%; an unrelated image pair produced 2,482
matches. More matches therefore did not establish more correct correspondences.

The reciprocal filter required mutual nearest endpoint pairs and a maximum
three-pixel Euclidean discrepancy at each endpoint. It retained 47,118 tiled
forward/reverse pairs, without improving the box proxy. In a post-hoc annotation
audit, 946 of 2,801 reciprocal source endpoints inside a drone box landed outside
the other drone box (33.8%). Only 22/32 directions had any reciprocal pair inside
both boxes. Both-box membership is not semantic or pixel accuracy: boxes include
background, and matches from overlapping tiles are correlated observations.

To test loss of local support during grid reduction, a separate candidate used
every match passing the unchanged local consensus test in piecewise affine
interpolation. It also passed only 1/16 pairs in each input variant. This changes
both control selection and interpolation, so it does not isolate the effect of
grid reduction. It provides no inverse or topology guarantee. Unsupported corners
remain failures. None of these diagnostics qualifies registration or authorizes
generator training.

Local report paths and SHA256 identities:

| Report under `experiments/` | SHA256 |
|---|---|
| `registration_antiuav_tiled_matching_gpu_01/report.json` | `1546737338c81b43628435db5a0fb3c3862c18759d08fdb8c5d712f858466733` |
| `registration_antiuav_tiled_reciprocity_01/report.json` | `3bab50627d2838bced7dc6341f6ca639d502b7f45b5a5c31afec7160949a6b6d` |
| `registration_antiuav_tiled_target_support_01/report.json` | `f5bab0b8cb5aee6199f75f4d2bc2a7db1421f98863ef47d7af9ee5e8b307d6b7` |
| `registration_antiuav_all_consensus_01/report.json` | `9e556dad639820c19f3708e1c26eb580bcaf0b97cc8855613184abf0e4e1acdb` |

The reports are local artifacts, not bundled public evidence. CPU reconstruction
requires the recorded source files, dataset inputs and saved match arrays:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.verify_antiuav_tiled_matching \
  --report experiments/registration_antiuav_tiled_matching_gpu_01/report.json
```

## MINIMA RoMa dense correspondence

MINIMA-RoMa was evaluated separately from the earlier MINIMA-XoFTR checkpoint
swap. The RoMa experiment uses RGB inputs, float32 inference, its fixed 560-pixel
coarse and 864-pixel fine grids, and direct dense coordinate queries without TPS
fitting or stochastic correspondence sampling. It is not a weights-only ablation
of the grayscale XoFTR pipeline. Saved dense fields were reconstructed on CPU;
this verifies scoring, not a repeat of neural inference.

On the Anti-UAV300 header-cropped train16 panel, the frozen certainty threshold
of 0.5 yields 0/16 joint box-proxy passes. A separate post-hoc diagnostic removes
only the confidence filter, retaining in-frame checks, and yields 5/16. This is
not a replacement acceptance gate. In the same-RGB known-shift control, only
12/608 queries survive the original threshold, whereas 607/608 ungated predictions
are within three pixels (median error 0.092 pixels). Low confidence therefore
cannot be interpreted as proof of an incorrect coordinate on that control.

External UAV-TIRVis author-provided landmarks test actual cross-modal pixel
errors. Inference receives images only; landmarks are used solely for scoring.
The four development pairs contain 241 corresponding points. All PCK values
below use a three-pixel threshold in the native thermal image, with unsupported
predictions retained as failures and each pair weighted equally.

| Method | Development macro PCK3 |
|---|---:|
| XoFTR with original local TPS | 83.72% |
| MINIMA-RoMa with certainty at least 0.5 | 41.24% |
| MINIMA-RoMa without confidence filtering, diagnostic only | 98.13% |

The ungated diagnostic places 238/241 development points within three pixels.
It does not establish a deployable confidence policy, physical correctness on
unlabelled pixels, or Anti-UAV300 qualification. The original gate remains HOLD.
Raw external images and labels are not licensed here for redistribution.

Development report: `experiments/registration_external_roma_gpu_01/report.json`,
SHA256 `b727e3aeea2cd03dc158a759590de3a9687fbe85aa8474ca17be644105fdb5f9`.
Anti-UAV report: `experiments/registration_minima_roma_gpu_01/report.json`,
SHA256 `53f123f1e03ac563585cd477ae7b4748af875c830bed05ca9d8b8c0dbedc367d`.

### Confirmation on additional images

The same frozen model and scorer were run on sample IDs 2, 3 and 4 from each of
the four scenes. These 12 pairs contain 499 authored correspondences; they are
additional images from known scenes, not an unseen-scene test. No model fitting
or threshold tuning used their landmarks. Both the original confidence branch
and the ungated diagnostic were retained.

| Method | All 12 pairs macro PCK3 | Excluding duplicate development labels |
|---|---:|---:|
| XoFTR with original local TPS | 63.37% | 61.71% |
| XoFTR with added consensus hull controls | 67.89% | 66.25% |
| MINIMA-RoMa with certainty at least 0.5 | 42.83% | 41.47% |
| MINIMA-RoMa without confidence filtering, diagnostic only | 96.54% | 96.35% |

For the ungated branch, 486/499 correspondences are within three native thermal
pixels and all are in-frame. Per-pair PCK3 ranges from 86.36% to 100%, so the
average must not be read as every image meeting a 95% criterion. The secondary
denominator is 11 pairs and 428 landmarks: `Seaside/2` has byte-identical RGB and
thermal landmark files to development sample `Seaside/1`. Excluding it does not
make the remaining images independent scenes.

This supports further evaluation of RoMa geometry and confidence calibration,
not automatic export of qualified generator-training pairs. Reverse-direction
errors are also stored in native RGB pixels; their numerical pixel thresholds
are not interchangeable with thermal thresholds because the resolutions differ.
The Anti-UAV300 registration and generator gates remain HOLD.

Confirmation report:
`experiments/registration_external_roma_confirmation_gpu_01/report.json`, SHA256
`14e31d0731f44f9b96b2468d19e26acdb9f07d5fa4d6f3d7d3e25e0bf7f2fcd5`.
All saved-field scores and both subset aggregates were reconstructed successfully
after inference. To verify an existing local result without running the model:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_external_roma_confirmation verify \
  --out-dir experiments/registration_external_roma_confirmation_gpu_01
```

### Further confirmation and failure rejection

Before reading new matching errors, sample IDs 5 and 6 from each scene were
selected for another fixed-model check. `Seaside/6` contains images but no authored
landmark files, so it was excluded for missing reference data before inference,
without replacement. The seven eligible pairs contain 363 landmarks.
`Seaside/5` repeats earlier landmark files; the secondary nonduplicate subset
contains six pairs and 290 landmarks. These remain previously observed scenes.

| Ungated RoMa measurement | All 7 eligible pairs | Excluding duplicate labels |
|---|---:|---:|
| Macro PCK3 in native thermal pixels | 85.64% | 83.48% |
| Landmark count | 363 | 290 |

`MountainResort/6` has PCK3 of 32.26%, median supported error 49.63 pixels and
95th-percentile error 201.02 pixels. The earlier 96.54% average must not be used
as a universal reliability claim. Saved scores and aggregates were reconstructed
successfully; this validates the reported failure rather than qualifying the model.

A subsequent **post-hoc** diagnostic retains a correspondence only when its
roundtrip error is at most one thermal-equivalent pixel and forward/reverse
sampled Jacobians are positive. The rule receives predicted maps and query
positions, not the reference destination. GT is consulted only to score retention.
On `MountainResort/6`, it rejects 41 of 42 points whose error exceeds three pixels,
retaining 21/62 points, of which 20 are correct. The conditional accuracy is
95.24%, but coverage is only 33.87% and all-landmark PCK3 remains 32.26%.
Across all seven pairs it retains 17 incorrect points and rejects no correct ones
in this sample. This is limited failure detection, not correspondence repair.
Perfectly invertible wrong transformations can still pass the rule. It has not
been confirmed on fresh data or established as a dense support mask.

Neither result changes Anti-UAV or generator authorization. Local evidence:

- `experiments/registration_roma_fresh_confirmation_gpu_01/report.json`, SHA256
  `808564329bd00e6684d5cd418713e5b64a242e9875693e0fa7b5bfb71f868739`.
- `experiments/registration_roma_cycle_rejection_checked_01/report.json`, SHA256
  `a3adbd94c05666214928901b12e6fd2f75917c91c7f6418dcb5cb4b7c9775973`.

### Geometric fallback and trained coarse alignment

Two image-map-only fallback rules were evaluated without fitting to reference
destinations. Both obtain controls from a fixed 64 by 64 image grid, retaining
points with positive forward/reverse Jacobians and at most one target-equivalent
pixel of roundtrip error. Local affine interpolation requires at least 12 nearby
controls and a query inside their convex hull. On the seven external pairs,
no rejected reference query met the full repair conditions, so accuracy did not
change. A separate global affine RANSAC fallback changes 41 predictions in the
largest failure case but increases PCK3 only from 20/62 to 21/62. On Anti-UAV,
the same global fallback leaves the ungated joint box count at 5/16; substituting
the global affine everywhere reduces it to 1/16. Neither fallback is adopted.

The subsequent GPU comparison combines the completed uniform300 image-conditioned
model with RoMa. The coarse model receives independently resized native RGB and
IR images at its original 256 by 256 input size. Its IR-to-RGB sampling map renders
RGB onto the IR grid. RoMa then matches that rendered image to the observed IR
image, and the residual maps are composed back into the original image domains.
Pixel-centre scaling, both directions, header offsets and missing rendered support
are handled explicitly. Neither rendering nor residual matching receives GT boxes;
boxes are consulted only for the following engineering proxy.

| Anti-UAV train16 condition | Both box IoUs at least 0.6 |
|---|---:|
| Trained coarse model with matched observed-support exclusions | 11/16 |
| Coarse model followed by RoMa, certainty at least 0.5 | 0/16 |
| Coarse model followed by RoMa, ungated diagnostic | 4/16 |

The ungated composition loses seven coarse-model passes and gains none. A
same-image known-shift control has 910/912 queries within three pixels (99.78%).
That control and passing coordinate tests do not establish RGB–IR correspondence.
This result rejects the tested composition, not every possible coarse-to-fine
matcher. The coarse model was trained on these sequences; this is not held-out
generalization, independent pixel GT or generator-training approval.

Inference completed on an RTX 3090, and all saved pair scores plus the known-shift
score were reconstructed on CPU. Original inputs, checkpoints and earlier reports
are unchanged. Local evidence:

- `experiments/registration_roma_local_repair_checked_01/report.json`.
- `experiments/registration_roma_global_repair_01/report.json`.
- `experiments/registration_antiuav_roma_global_repair_checked_01/report.json`.
- Coarse input manifest: `experiments/registration_roma_prewarp_inputs_01/manifest.json`,
  SHA256 `84b303302cc34d588504d5fb4ad56d050724570a8201036df8cfc55ebcbff706`.
- GPU comparison: `experiments/registration_roma_prewarp_gpu_01/report.json`,
  SHA256 `b33c04fd9eff28c53014dd194506b19dfb38a222214ded4bde3a9be06218e778`.

Saved-result verification does not rerun neural inference:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m scripts.probe_roma_prewarp_gpu verify \
  --out-dir experiments/registration_roma_prewarp_gpu_01
```
