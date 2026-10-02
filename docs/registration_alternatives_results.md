# Registration alternatives: executed diagnostics and remaining evidence

These are train-only, bounded diagnostics, **not successful registration qualification**.
No checkpoint, frozen geometry threshold, final test split or generator state was changed.
The two distinct blockers are accurate physical alignment and independent evidence of it.
Current `qualification_v4.py` intentionally holds generator eligibility until the latter
exists; changing that constant would not solve either blocker.

## XoFTR input header ablation

Report: `experiments/xoftr_overlay_train16_01/report.json` (162.9 seconds CPU).
Same16 train midpoint pairs and pinned weights as the previous detector-free diagnostic.
Original predictions were reused after source, input and NPZ hash verification. New
inference crops the top20%, rounded upward to eight rows: RGB72/360 and IR104/512.
No rescaling follows cropping; output y offsets are restored. This is a **header**
ablation, not removal of central reticles. Header regions may also contain real scene.

| Condition | Median matches | Frames with both-box matches /16 | >=4 unique reciprocal both-box pairs /16 | Also >=10% hull in both boxes /16 |
|---|---:|---:|---:|---:|
| Original input | 265.5 | 11 | 8 | 7 |
| Original input, header output matches removed | 41 | 11 | 8 | 7 |
| Header removed before inference | 23 | 11 | 7 | 7 |

One frame's target crosses the removed band; the report also provides a common-target
15-frame denominator. Both filtered conditions are evaluated in the same retained domain.
Homography box-IoU>=0.6 counts decrease from5/16 (postfilter) to3/16 (input crop);
conditional mean IoU is0.3925 vs0.3553, each available for13 frames. These are proxy
statistics, not landmark accuracy or evidence that every automatic match is correct.

On the fixed unrelated-sequence negative, matches decrease from170 original (32 after
header output filtering) to7 with input cropping. This supports an overlay/context
confound but does not isolate text removal from loss of other cropped context or changed
normalization. Known(+8,-8) RGB shifts have100% matches within3px on both control images.
**Header removal suppresses many misleading matches but does not solve UAV registration.**

Re-run, with a fresh output path:

```bash
AERO_DEVICE=cpu AERO_OVERLAY_OUTPUT="$PWD/experiments/xoftr_overlay_repeat01" \
  bash scripts/run_xoftr_overlay_screen.sh
```

On an allocated GPU use `AERO_DEVICE=cuda`; the wrapper preserves the scheduler's device
visibility. No GPU is needed to read or analyze the saved results. The exact executed
probe source is archived beside the report; current code differs only by line wrapping.

## Similarity objectives: MI and NGCC

Both agents tested the same frozen-v6 checkpoint, same160 train midpoint pairs and75
small transform corrections, with original and header-excluded scoring support.
Frozen-v6 **model inputs remain unchanged** here: only loss observations exclude headers.
Source GT boxes never select a correction; box IoU is computed afterwards as a proxy.

| Method, header-excluded expanded ROI | Eligible IR→RGB / RGB→IR | Mean Δbox IoU IR→RGB / RGB→IR |
|---|---:|---:|
| Gaussian-Parzen MI | 157 /157 | −0.05105 /−0.04776 |
| Hard-histogram MI reference | 157 /157 | −0.05289 /−0.04615 |
| Absolute gradient NCC | 157 /157 | −0.05200 /−0.04391 |
| NGF-style squared gradient cosine | 157 /157 | −0.07195 /−0.05421 |

These are **within-method changes from frozen v6**, not a controlled ranking between
methods: support/texture eligibility and MI's maximum4096 histogram samples differ.
The earlier MIND results likewise do not use identical observation masks. Same-panel
synthetic implementation checks recover320/320 known translations for MI and the
polarity-invariant gradient scores. Real exact-box checks also remain negative.
This does not support full training driven primarily by these objectives; combined
objectives and larger searches were not tested or ruled out.

Authoritative reports:

- `experiments/registration_mi_probe_train160_03/report.json` (206.9 seconds CPU);
  [MI definitions, controls and commands](registration_mi_probe.md).
- `experiments/registration_ngcc_train160_03/report.json` (354.8 seconds CPU);
  [NGCC definitions, controls and commands](registration_ngcc_probe.md).

Earlier numbered runs are retained, not promoted; some interrupted runs have no final
report. Reproduce both full panels under a fresh common directory with
`bash scripts/run_registration_alternatives.sh`. This is CPU diagnosis, not300-epoch training.

## Global transform fitted to actual matches

Unlike the older affine projection of predicted flow, this tests transformations fitted
directly to reciprocal XoFTR matches, without GT-based match selection.

| Transform | Eligible fits postfilter→input crop /16 | Both directions box IoU>=0.6 postfilter→input crop /16 |
|---|---:|---:|
| Similarity | 12→9 | 3→3 |
| Affine | 12→10 | 3→3 |
| Homography | 6→5 | 2→2 |

No tested global model resolves this panel. Analytic inversion does not prove correct
alignment, and conditional mean improvements on different eligible subsets are not
overall gains. See [geometry results](registration_match_geometry.md) and
`experiments/registration_match_geometry_01/report.json`.

## Frozen semantic features: DINOv2

Official frozen ViT-S/14 with register tokens was evaluated on the same16 pairs,
without fine-tuning, confidence tuning or GT-driven cropping. Report:
`experiments/dinov2_matching_train16_01/report.json` (83.8 seconds CPU).

| Whole-image condition | Any both-box patch matches /16 | >=4 both-box patch matches /16 | Also >=10% hull in both boxes /16 |
|---|---:|---:|---:|
| Original input | 12 | 1 | 1 |
| Header-cropped input | 12 | 0 | 0 |

Both conditions yield only17 total both-box patch matches. Fewer than four patch
centres fall inside one of the target boxes in9/16 original and8/16 cropped pairs.
Mutual nearest neighbours are mutual by construction, not independent reverse-replay
evidence. RGB color inputs and a14px token grid also differ from the XoFTR grayscale
fine matcher; do not equate their raw match counts or confidence scores.

Known28px patch-grid shifts have median error0 but only82.3–97.0% of interior matches
within3px across controls. These are model-grid units, not native camera accuracy.
The unrelated-pair control still produces36/32 matches (original/cropped), none both-box.
This whole-image token method is not a validated fine-scale anchor for tiny UAVs;
predicted-ROI crops and feature-loss training were not tested. See
[DINOv2 diagnostic](registration_dinov2_probe.md) and the
[official implementation](https://github.com/facebookresearch/dinov2).

## Remaining alternatives and unnecessary human prerequisites

- **Central HUD exclusion:** implemented raw-postfilter vs median/mean input fill
  under identical mask+8px exclusion support. [Mask manifest and command](xoftr_hud_masks.md).
  This older reviewed-mask variant was not executed. A separate automated thin-line
  heuristic plus native target-crop ablation now avoids human masks; see
  [automatic repair diagnostics](registration_automatic_repair.md). Heuristic masks
  are not relabeled as reviewed masks. Header cropping above is not complete HUD removal.
- **Silhouette Dice/Chamfer:** implemented fixed-support scoring of explicitly reviewed
  same-physical-outline masks and frozen candidate maps. [Input contract and command](registration_silhouette_probe.md).
  No reviewed masks exist yet, so no real-data score is claimed. Requiring human review
  before even testing an automatic SAM pseudo-mask objective was unnecessarily restrictive.
  The new SAM extraction/similarity probe uses automatic masks as noisy optimization
  signals, not independent GT. SAM v1 completed; the corrected native-coordinate
  cached-mask replay and fresh corrected-crop inference are also complete:
  see [SAM v2 results and command](registration_sam_v2.md). The subsequent
  [non-deterioration ablation](registration_sam_pareto.md) removes Chamfer regressions
  by retaining identity in the problematic cases; it is not new physical evidence.
- **Timing calibration, temporal aggregation and local invertible training:** actual
  pairing correction, aggregation and local invertible training remain unexecuted.
  The [shared-velocity transform primitive and CPU initialization diagnostic](registration_shared_velocity.md)
  are implemented; the naive initialization passes8/16 train frames geometrically
  but worsens box overlap. This is not learned local/invertible registration.
  An automatic train-box-trajectory lag screen has now been run (linked above).
  Equal frame indices/FPS do not prove synchronization; reproducible multi-frame
  timing evidence is needed before pairing changes, not necessarily human annotation.
  Global-transform failure alone is not a result for a local/invertible architecture.

Additional completed automatic diagnostics are recorded separately:
[external-reference/local TPS](registration_local_warp_diagnostic.md),
[RGB-resolution comparison](registration_rgb_resolution.md), and
[MINIMA-XoFTR checkpoint comparison](registration_minima.md). The latter improves
one external-reference configuration but does not increase Anti-UAV box-proxy
passes. These are executed alternatives, not proposed GPU training or qualified
pixel GT. None changes the original registration/generator HOLD.

## Historical human-review route: paused, not a repair prerequisite

**Both forms are paused.** The
[Korean six-pair familiarization](registration_practice_ko.md) was distributed as
`experiments/registration_part_practice_v1/practice_A.tar.gz` and `practice_B.tar.gz`.
These contain part diagrams, fixed choices, explicit uncertainty/visibility options
(not a numerical radius) and optional paired practice clicks. Responses are never
automatically treated as registration GT. The48-pair archives below are historical;
do not redistribute them as the current task.

`experiments/registration_physical_review_train48_02` contains48 native RGB/IR frame
pairs from the same16 train sequences, without model outputs or GT boxes on the images.
Open `review_A.html` and `review_B.html` independently on a local computer after copying
the directory. Each reviewer identifies real shared physical features, uncertainty,
visibility and unobservable cases; then exports their own JSON.

Distribute the separate archives `experiments/registration_review_A_train48_02.tar.gz`
and `experiments/registration_review_B_train48_02.tar.gz` (31.75MiB each). Each contains
the common images/manifest and only that reviewer's HTML/template. Preserve the
relative `images/` directory after extracting. Do not exchange the reviewers' answers
before both have completed their independent annotations.

[Review instructions and validation command](registration_physical_review.md).
Blank templates deliberately fail validation; the96 image hashes were verified.
This development panel measures observability and reviewer uncertainty. It is not the
untouched final scientific panel, and annotation structure validity is not qualification.

Required before release: semantic adjudication, independently frozen accuracy/coverage
policy, physical accuracy evaluation and unchanged geometry checks on the declared
supervision domain. Target-only evidence cannot authorize full-image paired supervision.
If common physical features are not observable, better synchronized/calibrated paired
data is needed; repeated optimization cannot manufacture those missing observations.
