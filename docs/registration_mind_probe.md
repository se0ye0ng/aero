# MIND registration diagnostic

## Question and scope

Does a local self-similarity descriptor supply a more useful alignment signal than
the existing structural edge NCC on Anti-UAV300 RGB/IR pairs? This is a bounded
loss-landscape test, not a trained MIND model or a registration qualification.
No optimizer steps, GPU jobs, validation/test reads, checkpoint changes or gate
relaxations are performed.

The implementation in `src/aero_ir/registration/mind.py` is a **2D MIND-style
adaptation**, not the original 3D MIND or MIND-SSC implementation. Reference:
[Heinrich et al., Medical Image Analysis 2012](https://doi.org/10.1016/j.media.2012.05.008).
RGB is converted to grayscale. Four axial neighbours at radii 1 and 2 each
produce uniform 3x3 patch SSDs; distances are divided by their local mean and
exponentiated, with maximum normalization per radius. The descriptor has eight
channels. Source/target descriptors are extracted before warping, using the
existing pixel-centre, `align_corners=False` geometry.

## Experimental controls

- Frozen v6 checkpoint: `b719c61b6ba991a1855c66ca32d7cbe1d86646498a289c5fc2d434ba3cf6f02e`.
- Train cache manifest and checkpoint train-only metadata are verified.
- Sequence selection is deterministic: evenly spaced sorted sequence IDs,
  midpoint usable cache pair. One pair per sequence is not exhaustive evaluation
  of all training frames. Cache positions are not original video frame numbers.
- Both point-map directions are measured separately. Source GT boxes are used
  only for the box-IoU diagnostic, not to choose the visual-loss minimum. Target
  boxes define diagnostic regions and the predicted enclosure defines the scale
  pivot. This is annotation-assisted analysis, not a label-free deployed system.
- MIND SSD and edge NCC use exactly the same fixed support for every candidate.
  Both descriptor footprints and the intersection of valid support across the
  candidate set are required. Flat patches (mean SSD <= 1e-6 in [0,1] units) are
  excluded for all candidates together; fewer than 32 supported pixels causes
  abstention. This texture cutoff is an engineering choice, not calibrated
  confidence. Missing/constant NCC observations also abstain.
- Target ROI is twice the annotated box extent with minimum 24px width/height;
  background is its complement. A separate sensitivity run uses only the exact
  annotated box (`--roi-mode box`), with no context expansion. Neither region is
  a segmentation mask or an independent physical correspondence measurement.
- Candidate translation units are 256x256 network pixels, not native pixels.
  Coarse search: {-8,-4,0,4,8}px in each axis, scales {0.9,1,1.1}.
  Fine search: {-2,-1,0,1,2}px, scales {0.98,1,1.02}.
  Both have 75 candidates including the unchanged native map.
- The diagnostic minimizes each visual loss separately. It does **not** test a
  weighted sum with box/cycle losses and cannot establish that every such sum
  will fail.

## Implementation verification

Tests cover known translation under contrast inversion, affine intensity
invariance, finite nonzero displacement gradients, flat-region abstention,
fixed support, empty support rejection, coordinate sign and source-GT isolation.
The real-image synthetic control warps a single modality by a known (+4,-4)px
backward map and inverts contrast; this is an implementation check, **not** a
simulation of all RGB/thermal appearance differences.

## Initial 16-sequence coarse diagnostic

Artifact: `experiments/registration_mind_probe_train16_01/report.json`.
CPU evaluation took 20.6 seconds after model loading. Both methods recovered
the known synthetic displacement on 32/32 controls (16 RGB and 16 IR images).

| ROI criterion | MIND IR->RGB | NCC IR->RGB | MIND RGB->IR | NCC RGB->IR |
|---|---:|---:|---:|---:|
| Eligible / evaluated | 15/16 | 15/16 | 16/16 | 16/16 |
| Mean box IoU change | -0.2326 | -0.1307 | -0.2046 | -0.1126 |
| Improve by >0.01 | 0 | 0 | 1 | 1 |
| Worsen by >0.01 | 12 | 13 | 13 | 10 |

The candidate set did contain a box-IoU improvement >0.01 for 9/16 samples in
each direction, so all-negative changes cannot be attributed solely to there
being no available better-box candidate. This oracle uses annotations only as a
diagnostic and is not a deployable selector.

## Expanded 160-sequence fine diagnostic

Artifact: `experiments/registration_mind_probe_train160_fine_01/report.json`.
All 160 official train sequences contribute one pair. This search was added after
observing the coarse result to check whether large candidate perturbations alone
explained the failure; it is exploratory, not an untouched confirmatory split.
CPU evaluation took 208.0 seconds after model loading. Both MIND and NCC recovered
the known shift on 320/320 synthetic controls. All real ROI scores had adequate
support under the stated fixed masks.

| ROI criterion | MIND IR->RGB | NCC IR->RGB | MIND RGB->IR | NCC RGB->IR |
|---|---:|---:|---:|---:|
| Eligible / evaluated | 160/160 | 160/160 | 160/160 | 160/160 |
| Mean box IoU change | -0.06770 | -0.04161 | -0.03893 | -0.03716 |
| Median box IoU change | -0.03251 | -0.01680 | -0.01745 | -0.01503 |
| Improve by >0.01 | 30 | 36 | 30 | 34 |
| Worsen by >0.01 | 106 | 94 | 100 | 93 |
| Within +/-0.01 | 24 | 30 | 30 | 33 |

The candidate set includes a >0.01 box-IoU improvement for 132/160 IR->RGB and
129/160 RGB->IR cases. Selecting the lowest visual loss usually does not identify
those candidates. Background-selected MIND corrections also decrease mean box
IoU (-0.09230 / -0.08871); background alignment is not target alignment evidence.

Restricting this saved candidate set to pure translations (scale exactly 1) still
gives negative mean IoU changes: MIND -0.06432 / -0.03650; NCC -0.03973 / -0.03395.
This uses the existing common support of all 75 candidates. The negative trend
is therefore not solely an effect of including scale changes in the search.

## Interpretation and decision

### Exact-box sensitivity check

Artifact: `experiments/registration_mind_probe_train160_box_01/report.json`.
The same 160 pairs, fine candidate grid, checkpoint and support policy were used,
but appearance scores were restricted to the annotated box, excluding the added
context. The two RGB->IR exclusions had fewer than 32 supported informative
pixels; they were not counted as successes. Synthetic controls again passed
320/320 (the same controls, not 320 additional independent examples).

| Exact-box criterion | MIND IR->RGB | NCC IR->RGB | MIND RGB->IR | NCC RGB->IR |
|---|---:|---:|---:|---:|
| Eligible / evaluated | 160/160 | 160/160 | 158/160 | 158/160 |
| Mean box IoU change | -0.06396 | -0.06786 | -0.04537 | -0.04225 |
| Improve by >0.01 | 30 | 31 | 35 | 35 |
| Worsen by >0.01 | 108 | 101 | 97 | 98 |
| Within +/-0.01 | 22 | 28 | 26 | 25 |

The negative mean change is not explained solely by including surrounding
background. MIND is not uniformly worse than NCC in every configuration, but
neither standalone minimum consistently improves this box proxy. Rectangular
boxes still include background, and physical silhouette differences remain a
possible limitation, not an established explanation from these measurements.

### Decision

**Do not launch a full MIND-primary training run on the strength of these results.**
This descriptor passes known-geometry implementation controls but its standalone
minimum is not a reliable target-box alignment signal in this frozen-v6 local
search. Even the existing edge NCC shows conflicts with box agreement. Adding a
visual loss does not by itself solve the correspondence-identifiability problem.

This does not prove that every MIND variant, another initialization, a confidence-
masked objective, or a small MIND term combined with geometry constraints will
fail. No such network training was conducted. Nor does a box-IoU decrease prove
every individual pixel got worse: independently reviewed landmarks/masks remain
missing. A combined-loss pilot would be a new experiment and should not be
represented as approved by this diagnostic. Keep original checkpoints and gates
unchanged; establish independent correspondence evidence before treating any
appearance-loss decrease as dense registration accuracy.

Fixed descriptor neighbourhoods are not guaranteed invariant to the real camera
scale/viewpoint differences. The known-translation/contrast controls do not test
all those effects. Conclusions apply to the stated 2D adaptation, local search
and frozen v6 maps; they are not a rejection of the entire MIND literature.

Verification: 11 new MIND tests and 66 combined registration regression tests
passed; Ruff and `git diff --check` passed. The only regression-test warning was
the existing PyTorch TF32 API deprecation warning.

## Reproduction

From the project root (CPU only; choose a new output directory for each run):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_registration_mind.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m scripts.probe_registration_mind \
  --cache-root experiments/antiuav300_registration_v2_full_train_cache \
  --checkpoint experiments/antiuav300_registration_v6_pilot_e10_seed0/antiuav300_registration_v6_e10.pth \
  --sequences 160 --search fine \
  --output-dir experiments/registration_mind_probe_train160_fine_repeat
```

For the exact-box sensitivity run, add `--roi-mode box` and choose another fresh
output directory. No GPU command is needed for these diagnostics.

Reports retain decoded input hashes, checkpoint/cache/source hashes, candidate
maps' parameters, every candidate score and box IoU, region support counts and
synthetic recovery results. No independently reviewed pixel correspondences are
available in this diagnostic. Generator eligibility remains **HOLD** regardless
of the script's exit status.
