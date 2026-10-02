# Gradient-vector registration diagnostic

This is a frozen-v6 train-only loss-landscape diagnostic, not trained NGCC,
independent pixel correspondence validation, or registration qualification.
No optimizer updates, validation/test access, checkpoint edits, or gate changes
are made. RGB and IR are compared in the existing 256x256 cache coordinates.

## Scores and coordinate handling

The existing `structural_edge_ncc` correlates warped Sobel **magnitudes**. This
diagnostic instead warps source grayscale first, then computes Sobel x/y gradients
in target coordinates. Simply warping vector channels would omit the coordinate
Jacobian under rotation/nonlinear transforms. Sobel kernels are divided by eight.

Three losses are evaluated separately:

- `signed_ngcc`: one minus normalized correlation of gradient vectors, after
  subtracting the spatial mean separately for x/y components.
- `absolute_ngcc`: one minus the absolute preceding correlation, accommodating
  a global contrast inversion but not arbitrary local polarity changes.
- `ngf_style_squared_cosine`: one minus average squared local cosine between
  gradient vectors, with denominator factors `norm_squared + 1e-6`. This is a
  specifically defined **NGF-style adaptation**, not a canonical NGCC formula.

Normalized-gradient multimodal registration has methodological precedent in
[Haber and Modersitzki](https://research.uni-luebeck.de/en/publications/intensity-gradient-based-registration-and-fusion-of-multi-modal-i/).
That precedent does not establish success on RGB/thermal UAV imagery.

## Protocol

- Same frozen v6 checkpoint as the MIND diagnostic, SHA256
  `b719c61b6ba991a1855c66ca32d7cbe1d86646498a289c5fc2d434ba3cf6f02e`.
- Train cache and checkpoint provenance verified; sorted evenly spaced sequence
  IDs, midpoint usable cache pair, both map directions. Selected arrays are hashed.
- Fine candidate grid: each translation axis `{-2,-1,0,1,2}` network pixels and
  scale `{.98,1,1.02}`; 75 candidates, including unchanged field. Scale pivot uses
  predicted target-box enclosure, not source GT. Source GT only evaluates IoU.
- Expanded target ROI, exact target box, and complementary background evaluated
  separately. These rectangles are not segmentation masks or pixel GT.
- Compare original support against header-excluded support: RGB upper 20%, IR
  upper 104/512. Headers are excluded from scoring with stencil-safe validity;
  no artificial masking edges are scored. This does not change the frozen v6
  field and is not equivalent to retraining v6 on cropped inputs.
- Identical valid support across **all** candidates, with margin for Sobel,
  and erosion of combined geometric/source-header validity in target coordinates.
  First retained source-pixel centers bound bilinear interpolation, preventing
  fractional sampling of excluded pixels. At least 32 pixels
  required; centered gradient energy in each image must exceed `1e-8`.
- No candidate-dependent texture masking. Invalid/constant scores abstain;
  an unchanged candidate wins numerical ties and an entirely flat landscape
  abstains. Thresholds are fixed exploratory engineering choices, not calibrated
  confidence. All eligible results are macro-averaged per sequence/direction.
- The scores select a candidate independently; no combined box/cycle loss is
  tested. A decrease in box IoU is a warning about this proxy, not proof that a
  physically correct pixel alignment got worse.

## Implementation checks

Eleven unit tests cover known translation, contrast reversal, constant-region
abstention, brightness/contrast invariance, finite nonzero displacement gradients,
rotation coordinate handling, empty/header support, mixed local polarity,
non-finite input rejection, candidate permutation, excluded-header perturbation
invariance, out-of-bounds-neighbor stencil erosion, and flat synthetic ties.

The initial `registration_ngcc_train16_01` report is superseded: audit tightened
bilinear header bounds and Sobel support, and removed truth-dependent synthetic
tie preference. Partial `train160_01`, `train160_02`, and `train16_02` runs were
intentionally interrupted before reports; they are not experimental results.
Authoritative current-code outputs use suffix `_03`.

## Final results

Artifacts: `experiments/registration_ngcc_train16_03/report.json` (36.2 seconds)
and `experiments/registration_ngcc_train160_03/report.json` (354.8 seconds CPU
after model loading). The 16-panel is a subset, not extra independent data.
All three methods recover the normal-contrast known shifts on 320/320 controls.
Absolute NGCC and NGF-style recover inverted-contrast shifts on 320/320; signed
NGCC recovers 0/320, consistent with its polarity sensitivity rather than a code
failure. Synthetic selection has no access to truth for tie-breaking.

Mean box IoU change after selecting each method's minimum, relative to unchanged
v6; direction order is **IR-to-RGB / RGB-to-IR**:

| Support and score | Expanded ROI | Exact box |
|---|---:|---:|
| Original, signed NGCC | -0.11445 / -0.12018 | -0.12104 / -0.12499 |
| Original, absolute NGCC | -0.05377 / -0.04319 | -0.05564 / -0.04240 |
| Original, NGF-style | -0.07157 / -0.05308 | -0.04871 / -0.04076 |
| Header excluded, signed NGCC | -0.11384 / -0.12236 | -0.12051 / -0.12730 |
| Header excluded, absolute NGCC | -0.05200 / -0.04391 | -0.05387 / -0.04312 |
| Header excluded, NGF-style | -0.07195 / -0.05421 | -0.04937 / -0.04303 |

Original expanded support has 160/160 eligible frames in both directions;
original exact-box support has 160/158. Header-excluded expanded support has
157/157, and exact-box support 157/155. Abstentions are not successes.
For header-excluded expanded absolute NGCC, improvement >.01 occurs in 31/39
frames versus worsening >.01 in 94/87. NGF-style improves 35/25 versus worsening
96/101. All corresponding medians are negative. Thus the negative result is not
just a few extreme outliers or solely due to contrast reversal. Removing the
header does not rescue this target-box signal. Background results are recorded
separately in JSON and are not evidence of target alignment.

**Decision:** do not start full NGCC-primary training from this evidence.
The bounded search does not prove all gradient-based objectives or weighted
combinations fail, and box IoU is not independent pixel GT. It establishes that
these explicitly defined standalone scores are not reliable local candidate
selectors for the current frozen-v6 target alignment. Registration and generator
qualification remain unchanged/HOLD.

## Reproduce

```bash
.venv/bin/python -m scripts.probe_registration_ngcc \
  --cache-root experiments/antiuav300_registration_v2_full_train_cache \
  --checkpoint experiments/antiuav300_registration_v6_pilot_e10_seed0/antiuav300_registration_v6_e10.pth \
  --output-dir experiments/registration_ngcc_train160_repeat01 \
  --sequences 160
```

Existing output directories are never overwritten. CPU execution uses one
PyTorch thread. Reports preserve every candidate score, IoU, support count,
selection, checkpoint metadata, input hash, and source hash for subsequent audit.
