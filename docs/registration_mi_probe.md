# Mutual-information registration diagnostic

## Scope

This is a train-only, frozen-v6 local loss-landscape diagnostic. It does not
train a registration network, access val/test, update checkpoints, or qualify
pixel correspondence. Generator eligibility remains **HOLD**.

The question is whether maximizing intensity mutual information selects a
locally more useful target alignment. Box IoU is an annotation-assisted proxy,
not independently reviewed physical pixel correspondence ground truth. A
negative standalone score-selection result does not prove that every combined
box/cycle/MI training objective must fail.

## Estimators and fixed protocol

`mutual_information.py` implements Gaussian-Parzen soft-histogram MI. This is
**not Mattes MI**: both marginals use normalized Gaussian bin memberships.
For paired samples the joint PDF is the average outer product of memberships;
MI is `sum(p_ab * log(p_ab / (p_a*p_b)))`. The diagnostic minimizes **negative
MI**, not joint entropy alone. Intensities are grayscale in [0,1]; 16 uniformly
spaced bin centres and sigma `0.5/15` are fixed before measurement. A 16-bin
hard-histogram estimator is a non-differentiable reference. There is no
candidate-wise intensity normalization or bin-count tuning.

Primary reference for histogram/Parzen MI and kernel-bandwidth limitations:
[ITK Software Guide, mutual-information registration](https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html).

- Frozen v6 checkpoint SHA256:
  `b719c61b6ba991a1855c66ca32d7cbe1d86646498a289c5fc2d434ba3cf6f02e`.
- Same deterministic train selection as the MIND diagnostic: evenly spaced
  sorted sequence IDs, midpoint usable cache pair, both map directions. The
  160 run covers all train sequences with **one pair each**, not every frame.
- The verified original training cache has 256x256 images. Translation units
  are cache/network pixels, not native camera pixels. Images are not the
  native-resolution inputs used by the XoFTR diagnostic.
- Exactly 75 candidate maps: dx,dy in {-2,-1,0,1,2}px and scale in
  {.98,1,1.02}; unchanged v6 is included. Source GT does not select candidates.
- Region scores are separated into expanded target ROI (twice box extent,
  minimum 24px width/height), exact target box, and expanded-ROI complement.
- Every candidate uses the same geometric-support intersection. Reuses the
  conservative 3px border footprint from MIND for comparability; MI itself has
  no descriptor footprint. At most 4,096 evenly spaced flattened support
  indices are used, identically for every candidate. No random resampling.
- Fewer than 128 samples causes abstention. Target standard deviation or any
  candidate's sampled-source standard deviation <=1e-3 causes whole-region
  abstention. These are prospective engineering safeguards, not calibrated
  confidence. Tied/constant score landscapes abstain; a tied baseline is kept.
- Original support is compared with top-band exclusion: RGB 72/360=20%, IR
  104/512=20.3125%, mapped to cache coordinates. Source sampling must reach the
  first accepted pixel centre so bilinear interpolation cannot touch excluded
  rows. Every candidate uses the common retained subset. Pixels are **excluded
  from statistics**, not filled with zeros and scored. Bands can remove actual
  scene pixels; central reticles and other overlays are not removed.
- The frozen v6 predictor still sees original inputs in both conditions. Only
  the score's observations change. This is not the same intervention as removing
  overlay inputs from XoFTR, which changes the matcher's attention.

## Implementation verification

Unit tests verify normalized joint PDFs, symmetry, nonnegativity tolerance,
agreement of hard histograms with independent NumPy calculations, finite
nonzero subpixel displacement gradients against finite differences, known-shift
recovery under identity/contrast inversion/gamma transforms, shuffled-pixel
negative control, flat/empty support abstention, source-header interpolation
exclusion, candidate-order support invariance and source-GT selection isolation.

Real-image synthetic controls translate each sampled single-modality frame by
(+4,-4)px and apply identity, contrast inversion, or gamma2. These are
implementation controls, **not a simulation of true thermal appearances**.
The six method/control combinations reuse the same image evidence and must
not be reported as six independent datasets.

## Reproduction

CPU only; no GPU command or training job is required. Choose fresh directories:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_registration_mi.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m scripts.probe_registration_mi \
  --cache-root experiments/antiuav300_registration_v2_full_train_cache \
  --checkpoint experiments/antiuav300_registration_v6_pilot_e10_seed0/antiuav300_registration_v6_e10.pth \
  --sequences 160 --output-dir experiments/registration_mi_probe_train160_repeat
```

Reports retain selected input array hashes, cache/checkpoint/source hashes,
every candidate score and box IoU, support counts, abstention reasons and
synthetic control outcomes. A completed script is not a qualification PASS.

The initial `_train16_01` report predates tightening the bilinear source-header
boundary. `_train160_01` was interrupted before report generation for that
correction. `_train16_02` predates correcting the synthetic selector's tie
baseline from the known truth to identity; `_train160_02` was interrupted for
that correction. Earlier directories are preserved, not promoted results;
use the completed `_03` reports. The real candidate selector always used identity.

This run includes Gaussian/hard MI reference estimators, not same-support MIND
or edge-NCC scores. Earlier MIND scores have different texture exclusions,
sample counts and eligibility; do not rank the methods directly by their
unpaired aggregate IoU deltas. A paired common-support comparison would be a
separate experiment. This diagnostic asks whether each MI estimator by itself
improves the unchanged v6 box proxy under its explicitly stated protocol.

## Completed 16-sequence engineering screen

Artifact: `experiments/registration_mi_probe_train16_03/report.json`.
CPU time was 18.67s after model loading. Both estimators recover all 32/32
known shifts under each of identity, contrast inversion and gamma2. Synthetic
selection prefers identity on ties, never the known answer.

Header-excluded Gaussian MI, expanded ROI, has mean box-IoU delta
**-0.07937 IR->RGB / -0.06411 RGB->IR**. All 16 pairs are eligible in both
directions. Improvements >0.01 occur in 3/16 and 1/16, respectively; degradations
>0.01 occur in 12/16 and 13/16. Implementation success therefore does not imply
that maximizing MI supplies a useful target-box correspondence signal.

## Completed 160-sequence diagnostic and decision

Artifact: `experiments/registration_mi_probe_train160_03/report.json`.
CPU time was **206.94s** after loading. Both estimators recover **320/320**
known shifts for each of identity, inversion and gamma2. All source hashes
verify, and selected input array hashes exactly match the 160-sequence MIND
fine diagnostic. There are 13 new passing MI tests (24 with MIND regression
tests); Ruff and `git diff --check` pass.

Mean box-IoU change relative to unchanged v6, IR->RGB / RGB->IR:

| Region and observation condition | Eligible pairs | Gaussian MI | Hard histogram MI |
|---|---:|---:|---:|
| Expanded ROI, original | 160 / 160 | -0.05094 / -0.04611 | -0.05190 / -0.04445 |
| Expanded ROI, header excluded | 157 / 157 | -0.05105 / -0.04776 | -0.05289 / -0.04615 |
| Exact box, original | 148 / 119 | -0.06051 / -0.03504 | -0.05423 / -0.03524 |
| Exact box, header excluded | 146 / 117 | -0.06101 / -0.03543 | -0.05476 / -0.03563 |
| Background, original | 160 / 160 | -0.12548 / -0.11349 | -0.13088 / -0.11656 |
| Background, header excluded | 160 / 160 | -0.15006 / -0.14123 | -0.13878 / -0.13529 |

Header-excluded expanded Gaussian MI improves >0.01 in **39/157 / 34/157**
pairs and worsens >0.01 in **85/157 / 95/157**. The search does contain a
better-box candidate (>0.01) in 132/160 / 129/160 cases, so failure is not just
absence of better-box alternatives. The source-annotation oracle is only a
diagnostic; MI selection never sees those oracle IoUs.

**Do not promote this MI objective to a full training run from this evidence.**
Both estimators work on known geometry yet their local score optimum tends to
degrade the target-box proxy. Header exclusion and exact-box restriction do not
reverse that conclusion. This is a result for these fixed intensity estimators,
the current v6 neighbourhood and one frame per train sequence—not a rejection
of every MI variant or every combined-loss model. No independent physical
correspondence qualification or generator release is established.
