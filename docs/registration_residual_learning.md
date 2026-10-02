# Image-conditioned residual-velocity feasibility

The completed replay/output-preservation CPU results and the next matched-budget
300-additional-epoch GPU sampling comparison are documented in
[Uniform versus failure-aware residual fine-tuning](registration_replay.md).
Both original GPU pilot checkpoints below remain unchanged; diagnostic heads
are not promoted to qualified models or used to initialize that comparison.

## Question and scope

The completed v7 pilots reached114/160 and115/160 train-midpoint joint passes,
below the unchanged95% candidate threshold. A separate annotation-assisted
per-frame optimization repaired three of four selected alignment failures but
could not repair one support/bounds failure. That test used each frame's boxes
during optimization and was not an image-conditioned registration method.

The next diagnostic asks whether **one shared network**, using images rather than
annotation-derived transforms, can learn a residual. It does not assume that a
successful box fit establishes fine-grained RGB/IR correspondence.

## Implemented candidate

`src/aero_ir/registration/residual_velocity.py` freezes the completed v7 geometry
predictor. A small convolutional head receives14 channels: visible image, infrared
image, visible image warped by the frozen forward map, base velocity, warped
source-support mask and centre coordinates. These are image/geometry-derived
inputs; the predictor accepts no boxes, pseudo-masks, sequence IDs or per-frame
optimized control targets.

Four stride-two convolution/GroupNorm/SiLU blocks predict a16x16 two-channel
control grid for256x256 images. Bilinear interpolation, tanh bounding to24 network
pixels and a fixed boundary taper produce a residual stationary velocity. The
final maps are jointly integrated from `base_velocity + residual` and its negative
using the existing seven-step routine. The output layer starts at zero, preserving
both base maps exactly before fitting. Shared velocity does not itself guarantee
observed support, positive discrete Jacobians or physical correspondence.

Existing train boxes supervise the unchanged v7 geometric training objective.
The baseline predictor is frozen; only the residual head receives optimizer
updates. MIND is not added in this diagnostic. The choice tests the alignment
mechanism; it does not establish that MIND is useless in other configurations.

## CPU diagnostic protocol

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_residual_learning \
  --out-dir experiments/registration_residual_learning_train8_probe8_01
```

- Start from the verified completed v7 geometry control.
- In each saved pass/fail stratum, the first four lexicographic train sequence
  IDs fit the head; the next four form a separate train probe. Total:8 fit and8
  probe midpoint observations. This deliberately selected panel is not a
  dataset-wide performance estimate.
- 200 full-batch AdamW updates, seed0, learning rate1e-4, weight decay1e-5;
  gradient norm clipping5. No best-iteration selection or probe-based early stop.
- Images/base contexts are cached in CPU memory, with selected input bytes
  checked against the completed v7 screen. No dataset pixels are overwritten.
- Boxes enter the fit loss and before/after measurements, never prediction.
  The fit function rejects probe-role observations.
- Score the final iterate with all unchanged bidirectional predicates; verify
  checkpoint reload reproduces the final CPU metrics.

The probe observations are withheld **only from this head's optimizer**. Their
sequences were used by v7, and all observations are from the official train split.
This is not validation/test performance, physical-GT verification, a completed
pilot, or the planned300-epoch experiment. The saved `diagnostic_head.pth` is
explicitly marked as a tiny-panel diagnostic and cannot be loaded as a completed
v7 checkpoint. No generator maps or qualified-pair manifests are exported.

## Verification and next decision

The CPU experiment above completed, with final checkpoint reload reproducing
the per-frame CPU measurements. Artifacts are in
`experiments/registration_residual_learning_train8_probe8_01/`.

| Deliberately selected train observations | Before | After200 head updates |
|---|---:|---:|
| Eight observations used to fit the shared head | 4/8 joint passes | 6/8 |
| Eight observations not used to fit this head | 4/8 joint passes | 5/8 |

The fit loss decreased from0.5000 before the first update to0.1748 before the last
update. This demonstrates actual image-conditioned learning in the implemented
path, not qualification or an established generalization gain. In particular,
the table is **not** directly comparable to the160-frame v7 rates, and neither
row is an independent held-out test. The first run's executed diagnostic sources
are retained in its `sources/` directory; later runner edits add automatic source
snapshots and shorten a limitations string without changing that run's evidence.

Tests check exact initial-map preservation, frozen base parameters/mode, gradients
to the residual head, dependence on image inputs, detached context, input-range
validation and rejection of probe observations from fitting.

A positive tiny-panel fit only justifies further investigation of image-conditioned
residual learning. It does not justify selecting favorable frames, relaxing
qualification thresholds, starting a paired generator, or publishing an efficacy
claim. A full candidate needs a frozen train/validation protocol, comparison with
the same base model and budget, independent physical evidence on its actual
supervision domain, and downstream detector evaluation before project completion.

## GPU matched pilot commands

**Completed result:** both GPU arms finished and the saved-product comparison
was independently rerun against the files in this workspace. Both have exactly
3,200 retained optimizer updates, one attempt and no discarded logged steps.
Source snapshots, checkpoints, completion records and training/screen identities
pass the verifier. This is saved-artifact verification, not a GPU training replay.

| Same160 train midpoint observations | Joint passes | Pass rate |
|---|---:|---:|
| Common completed-v7 initialization | 114 | 71.250% |
| Predictor continuation,10 additional epochs | 115 | 71.875% |
| Residual head,10 additional epochs | 117 | 73.125% |
| Unchanged candidate threshold | 152 | 95.000% |

Relative to initialization, continuation gains two passes and loses one;
the residual head gains six and loses three. In the paired final comparison,
112 observations pass both, five pass only the residual model, three pass only
continuation, and40 fail both. The net residual gain is **two observations,
1.25 percentage points**, not a solution to registration qualification. At least
35 additional joint passes are needed even for the residual model's train screen.
These train-only, single-seed results and different learning rates/capacities do
not establish independent physical alignment or a reliable architecture benefit.

Authoritative comparison:
`experiments/registration_residual_pilot_e10_seed0/comparison.json`.
Final checkpoint SHA256 identities:

- continuation: `63c6b32eb8b4d7406cc10207df3bdba4795db1a2d65fd686bc811b316b956afe`;
- residual head: `86ba5a3d068645a3275934ed9572bd8502075e32b477b2ed9527d05014e89dcb`.

Do not rerun the launch commands below against these completed output directories.
Neither candidate currently authorizes the next held-out engineering screen,
paired-generator training, or an automatic300-epoch scale-up.

CPU failure attribution was run on both final screens, rehashing the selected
cached frame/box bytes for the target-size diagnostic:
`experiments/registration_continuation_final_constraints_01.json` and
`experiments/registration_residual_final_constraints_01.json`.

| Failed predicate in either direction (overlapping counts) | Continuation | Residual head |
|---|---:|---:|
| Any joint criterion | 45 | 43 |
| Box IoU | 30 | 31 |
| Absolute area-ratio change | 28 | 29 |
| Centroid shift | 5 | 6 |
| Global/ROI cycle-error magnitude bounds | 0 | 0 |

Higher joint pass does not mean every component improved. The two arms fail on
different observations, with overlapping predicates. Remaining failures chiefly
concern target alignment; a smaller round-trip error alone does not address them.
For the residual head, only8/21 observations whose smaller RGB/IR annotated short
side is below8 network-input pixels pass, versus109/139 at least8 pixels. These
post-hoc size strata are on the256px input grid, not native camera pixels, and
are an association rather than proof that higher resolution solves alignment.

### Actual checkpoint visuals and native-image checks

`scripts/render_registration_residual.py` renders the first two lexicographic
IDs in each **paired final** stratum: residual-only passes, residual regressions,
both failing and both passing. All eight were actually rerun on CPU from the
completed checkpoints. Output: `experiments/registration_residual_visuals_01/`.
Its report records both directional CPU measurements, source/checkpoint identities
and16 output-image hashes. The cases are diagnostic selections, not a representative
estimate and not selected for favorable appearance.

Each main panel shows input RGB, observed IR, continuation RGB warp and residual
RGB warp with dataset boxes and target zooms. The companion `_native.png` shows
unwarped native RGB/IR target windows. These windows are separate annotated crops,
not a registered pair or physical correspondence GT. Original video frame indices
were recovered from the cache midpoint positions; resized native images and boxes
exactly reproduced the cached observations. Native RGB byte hashes are recorded.

The eight CPU pass/fail outcomes reproduce their saved strata. For example,
`20190925_133630_1_4` changes IR-to-RGB box IoU from0.5953 to0.6070, crossing the
0.6 bound, whereas `20190925_194211_1_7` regresses from0.6325 to0.5901.
The latter's native RGB crop shows mainly two colored light spots while native
IR shows a body-like outline. Thus at least this example's appearance mismatch
is not created solely by resizing to256px. This is visual interpretation of a
specific example, not a finding that all failed sequences lack physical matches.
Neither matching boxes nor fitting one observed silhouette to another establishes
that identical physical surfaces have been aligned.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.render_registration_residual \
  --out-dir experiments/registration_residual_visuals_repeat01
```

Five new synthetic tests check native/cache correspondence and reject changed
pixels, boxes, annotations and midpoint positions. The existing three renderer
tests still pass. Images remain under git-ignored `experiments/`; this does not
grant permission to redistribute the dataset images in a public release.

### Known source-translation response: executed CPU diagnostic

To test whether the models ignore actual image position changes,
`scripts/probe_registration_translation_response.py` applies four exact integer
RGB shifts: (+4,0), (-4,0), (0,+4), (0,-4) input pixels. IR is unchanged. It uses
the same eight preselected train cases above and both completed checkpoints;
there are no optimizer updates, landmark annotations or checkpoint selection.

For each shift d, the expected forward sampling map is the **model's own** original
map translated by d in source coordinates. Error is measured on fixed support
from that reference and the known transform, not a mask chosen by the new
prediction. The IR annotation defines the diagnostic target region, not a model
input. A projected response gain of1 follows the requested source translation;
0 means the prediction did not move. It is not a physical-accuracy score.

Report: `experiments/registration_residual_translation_response_01.json`.

| Model | Measured target regions | Median of per-region median error | Median projected response gain |
|---|---:|---:|---:|
| Predictor continuation | 32/32 | 0.6469 input pixels | 0.9956 |
| Residual head | 32/32 | 0.6205 input pixels | 0.9669 |

On this selected panel, neither model ignores source translation. That narrows
the repair hypotheses: a completely position-insensitive predictor is not supported
by these results. It does not prove correct original alignment, size/shape response,
fine physical correspondence or improvement on unseen scenes. The perturbation
also moves camera HUD and introduces black borders; context effects remain.
Five analytic CPU tests verify translation sign/units, the ideal/unchanged-map
references, and that invalid predictions cannot hide errors by shrinking support.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_translation_response \
  --out experiments/registration_residual_translation_response_repeat01.json
```

### Known source-scale response: executed CPU diagnostic

The translation result does not test target size. A separate, actually executed
diagnostic, `scripts/probe_registration_scale_response.py`, uses the same eight
preselected cases and both completed checkpoints. RGB alone is scaled about its
train-box centre, with IR unchanged. Seven fixed conditions are identity, isotropic
1.25/0.8, width-only1.25/0.8 and height-only1.25/0.8. Images use inverse-transform
bilinear resampling on the same pixel-centre lattice. The expected forward map
is the known affine transform of the model's **own original** sampling map.

Report: `experiments/registration_residual_scale_response_01.json` (112 records,
zero parameter updates, CPU only). Identity resampling has median target-region
error0 for both models. All eight regions are measurable under every condition;
box-size ratios use seven cases because `20190925_101846_1_1` lacks full
reference/expected box sampling support. This exclusion is recorded, not treated
as a pass. The new prediction cannot change the support used for measurement.

| Applied source-image scale (x,y) | Continuation median predicted box-size ratio (x,y) | Residual median predicted box-size ratio (x,y) |
|---|---|---|
| (1.00,1.00), identity control | (1.0000,1.0000) | (1.0000,1.0000) |
| (1.25,1.25) | (1.1820,1.0828) | (1.1664,1.0883) |
| (0.80,0.80) | (0.8895,0.8441) | (0.8705,0.8366) |
| (1.25,1.00) | (1.1600,1.0346) | (1.1598,1.0325) |
| (0.80,1.00) | (0.8750,0.9914) | (0.8801,0.9942) |
| (1.00,1.25) | (1.0224,1.0657) | (1.0344,1.0680) |
| (1.00,0.80) | (1.0350,0.8833) | (0.9692,0.8631) |

The ideal ratio equals the applied scale, not1. Under-response is apparent on
this selected panel, especially height expansion:25% applied becomes about7%
predicted. This supports investigating scale response, not a claim that it causes
every native alignment failure. Interpolation, tiny objects, HUD and full-image
context changes remain confounders. These numbers are neither physical accuracy
nor evidence that a repaired model has been achieved. Thirteen analytic tests
cover ideal/unchanged-map references, non-square coordinate units, inverse raster
transport, invalid scales, and fixed support under invalid predictions.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_scale_response \
  --out experiments/registration_residual_scale_response_repeat01.json
```

### Candidate training constraint: gradient viability, not a completed repair

`src/aero_ir/registration/scale_equivariance.py` implements a new experimental
loss without modifying either completed pilot's loss or source snapshots. For
the augmented RGB prediction it penalizes source-coordinate deviations from the
known transform of a **detached original** forward map. The per-observation loss
is mean SmoothL1 over xy errors in pixels within fixed supported target regions;
observations with no measurable region raise rather than contribute zero. Train
boxes define augmentation centres and loss regions, never model prediction inputs.

This is an equivariance regularizer, **not new cross-modal correspondence GT**.
It cannot correct an arbitrary wrong original map by itself: even a map collapsed
to the scaling centre can satisfy it. Original geometric supervision and unchanged
qualification must therefore be retained in any experiment using this loss.

An actual CPU backward diagnostic in the completed residual head covers the eight
cases and six nonidentity conditions, with **no parameter updates**. Report:
`experiments/registration_residual_scale_gradient_01.json`.

- All48 conditions produce finite head gradients.
- For the explicit candidate weight0.1, the median norm ratio of weighted scale
  gradient to original geometric gradient is0.4251.
- Gradient cosine median is−0.1534:21 positive,27 negative. A negative cosine
  indicates local conflict between the two objectives, not guaranteed long-run harm.
- Three further analytic tests check teacher detachment, useful student gradients,
  loss reduction by a synthetic-field step, fixed support and invalid-input rejection.

This justified a matched CPU optimization check, now completed below. It does
not justify increasing the weight until a training gate passes, altering
thresholds, or starting a300-epoch/GPU generator run. An improvement in synthetic
consistency alone does not qualify registration. No new GPU pilot has been run
or approved by the gradient diagnostic.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_scale_gradient \
  --out experiments/registration_residual_scale_gradient_repeat01.json
```

### Matched CPU learning and full-train screen: completed, not an adopted repair

`scripts/probe_registration_scale_learning.py` actually fitted two independent
copies of the completed residual head. Each received24 AdamW updates, batch4,
learning rate1e-4, weight decay1e-5 and gradient clipping5. The fixed scale-loss
weights were0 (control) and0.1 (candidate); all other settings and input identities
were identical. Both executed the same augmented forward passes. Their shared
data/scale schedule hash is `6b4c3f77e8c1eae7d4827f5e49c6f49c2d3fd5dca410901b4ffe371b034c06dd`.
The original forward map is detached afresh at each update, not fixed to the
initial checkpoint and not labeled as correspondence truth.

The first sorted train ID in each of the four existing paired pass/fail strata
was used for fitting; the second was a probe only. This is four fit/four probe
observations, all previously used by the main training pipeline. No validation
or test split was accessed. Frozen image contexts were cached; neither arm
updated the base predictor. The final iterate was saved without best-iteration
selection, and reloading each diagnostic head exactly reproduced its CPU metrics.

Report: `experiments/registration_scale_learning_train4_probe4_01/report.json`.

| Four observations per role | Initial | Geometry only | Geometry + scale0.1 |
|---|---:|---:|---:|
| Fit original-image joint passes | 2/4 | 2/4 | 2/4 |
| Probe original-image joint passes | 2/4 | 2/4 | 2/4 |
| Fit scale-response error, px | 0.8246 | 1.0037 | 0.8995 |
| Probe scale-response error, px | 1.1155 | 1.3737 | 1.0195 |
| Fit mean original geometry loss | 0.4181 | 0.2891 | 0.3011 |
| Probe mean original geometry loss | 0.3888 | 0.4202 | 0.4058 |

Scale errors are medians of24 per-region median errors (four cases times six
conditions). All models are measured on the **same initial-reference support**,
so a newly invalid map cannot improve its score by reducing the measured region.
Current source-support fractions are also retained. The equal probe pass counts
hide one gained and one lost observation in both arms; individual outcomes are
recorded. The scale term improves the probe consistency diagnostic relative to
the control but does not improve original-image joint pass count.

The saved heads were then actually evaluated on **all160 original train midpoint
pairs**, without further fitting, by `scripts/screen_registration_scale_learning.py`.
The initial head's CPU pass/fail identities exactly match the saved GPU screen.
Report: `experiments/registration_scale_learning_train160_screen_01/report.json`.

| Full train-midpoint screen after the tiny fit | Passes | Rate | Gains vs initial | Losses vs initial |
|---|---:|---:|---:|---:|
| Completed residual pilot, unchanged | 117/160 | 73.125% | 0 | 0 |
| Geometry-only tiny fit | 101/160 | 63.125% | 4 | 20 |
| Geometry + scale0.1 tiny fit | 111/160 | 69.375% | 4 | 10 |

Excluding the four fit observations yields115/156 initially,99/156 for control,
and109/156 for the candidate. These are **not independent held-out accuracy**:
all sequences participated in the main training. They do show that fitting a
small selected panel regressed other train observations. Scale regularization
reduced that regression in this run, but the candidate remains below its starting
model. **This configuration is not adopted as a registration repair, and no
full GPU run or generator training is approved by these results.**

A follow-up comparison covers all train sequences uniformly rather than repeatedly
fitting four selected frames; its completed results are below. Original-image
joint criteria remain unchanged. Improved synthetic equivariance alone remains
insufficient. The diagnostic checkpoints have a separate type and cannot be
loaded as completed full-training pilots.

Six added tests cover deterministic disjoint role selection, rejection of probe
data from optimization, real optimizer updates and matched schedules, fixed
evaluation support, paired gains/losses and diagnostic-type/budget rejection.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_scale_learning \
  --out-dir experiments/registration_scale_learning_train4_probe4_repeat01

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.screen_registration_scale_learning \
  --diagnostic-root experiments/registration_scale_learning_train4_probe4_repeat01 \
  --out-dir experiments/registration_scale_learning_train160_screen_repeat01
```

### Balanced train160 CPU sweep: completed, no clear scale-loss benefit

`scripts/probe_registration_balanced_scale_learning.py` actually trained two new
copies of the **original completed residual pilot**, not either tiny-fit head.
Each of the160 train sequence midpoints was used exactly once, in the same
seed0-shuffled order in both arms:40 AdamW updates of batch4, learning rate1e-4,
weight decay1e-5, gradient clipping5, scale weights0 and0.1. Six known source
scales rotate by batch; the full schedule is recorded under SHA256
`c58e0743656c469c9b89e039ac111a553f71bdc69248461f5ad62adefed6392a`.
All160 observations in each arm had measurable auxiliary regions. The runner
nevertheless explicitly records missing auxiliary evidence if encountered; it
never drops such observations from the original geometry objective or calls them
zero-error measurements. Both arms update only the residual head.

Each sequence's cached quarter-position frame, `floor(pairs/4)`, is a second
fixed panel excluded from these updates. These are **not independent validation
or test observations**: both panels are from sequences used in earlier training.
The source predictor and initial head remain frozen throughout. All used cache
bytes were rehashed, and the initial CPU midpoint pass/fail identities reproduced
the saved GPU result. Both final checkpoints contain exactly40 optimizer steps;
their tensors roundtrip exactly on reload. This is not a GPU training replay.

Actual report and checkpoints:
`experiments/registration_balanced_scale_train160_sweep1_01/`.

| Unchanged joint criteria | Initial | Geometry only | Geometry + scale0.1 |
|---|---:|---:|---:|
| Midpoint frames used in this sweep | 117/160 (73.125%) | 118/160 (73.750%) | 118/160 (73.750%) |
| Quarter-position frames excluded from this sweep | 115/160 (71.875%) | 114/160 (71.250%) | 115/160 (71.875%) |

On midpoints, both arms gain exactly the same two observations and lose the same
one. On quarter-position frames the scale arm retains the initial outcomes,
whereas the control loses one pass. This is a marginal preservation effect, not
evidence that scale consistency solves registration. The remaining midpoint
failures still include29 area-ratio violations and30/31 IoU violations for the
scale/control arms (overlapping counts). No thresholds were changed.

The large regression from repeatedly fitting four cases did not recur in this
balanced sweep. That cross-experiment comparison also changes update count and
observation exposure, so it does **not** isolate a causal sampling effect. Within
the balanced comparison, data, updates and initialization are matched. Its result
does not support promoting scale regularization to the main repair or launching
a long GPU experiment solely on this basis. Neither the95% engineering threshold
nor independent physical correspondence qualification has been met.

The existing completed GPU loss traces were also inspected. Residual-head mean
training total loss decreases from0.391591 in epoch1 to0.365303 in epoch10;
mean box-IoU loss decreases from0.306786 to0.288878. Updates with pre-clip gradient
norm above5 fall from22.81% to3.12%. This does not indicate universally missing
gradients or clipping every update. However, epoch samples rotate, so these means
do **not** establish convergence on a fixed panel or prove that longer training
would solve the failures. The next diagnostic should distinguish inability to
fit a known alignment-only failure from failure to learn a shared image-based
correction, before adding more auxiliary losses.

Four tests cover complete/disjoint panel construction, exactly-once balanced
sampling, explicit treatment of missing auxiliary evidence, and rejection of
nonfinite inputs. The full594-test CPU suite and repository lint pass.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_balanced_scale_learning \
  --out-dir experiments/registration_balanced_scale_train160_sweep1_repeat01
```

### Fit-capacity sanity check: four failures can be fitted, but transfer regresses

`scripts/probe_registration_fit_capacity.py` selects the first alignment-only
failure for each distinct filename prefix, then the first four prefixes in sorted
order. Other engineering predicates must already pass. Prefix diversity avoids
selecting only adjacent numbered clips, but is **not** proof of independent
recordings. Selection is based on the completed residual pilot's train screen;
the four initial failures were confirmed by fresh CPU inference.

Each independent head fits one selected frame for200 AdamW updates; a fifth,
shared head fits all four as batch4 for200 updates. Every head starts from the
same completed residual checkpoint. The original predictor, inputs, coordinate
convention, v7 geometry loss, learning rate1e-4, weight decay1e-5, gradient clip5
and qualification predicates are unchanged. No scale/MIND loss is added. There
is no early stopping or best-iterate selection. Every fitted observation is
presented200 times, but four independent heads have greater combined capacity
and optimizer-update count than one shared head: this is a fit-capability test,
not a compute-matched architecture benchmark or deployable per-case routing.

Actual report: `experiments/registration_fit_capacity_train4_01/report.json`.
All final checkpoints reload to exactly the same CPU measurements.

| Selected train observation | Initial joint pass | Independently fitted head | Shared four-frame head | Shared final IoU (IR→RGB / RGB→IR) |
|---|---|---|---|---|
| `20190925_101846_1_3` | No | Yes | Yes | 0.9739 /0.9738 |
| `20190925_130434_1_3` | No | Yes | Yes | 0.9867 /0.9545 |
| `20190925_133630_1_5` | No | Yes | Yes | 0.9856 /0.9841 |
| `20190925_140917_1_2` | No | Yes | Yes | 0.9828 /0.9835 |

This demonstrates that the existing image-conditioned head and loss can satisfy
all current engineering predicates on **these four fitted observations**. It
does not establish capacity on all160 sequences, fine physical correspondence,
or generalization. In particular, the nearly unit box IoUs are same-frame fits
to weak annotations and must not be presented as physical registration accuracy.

The **one shared head**, with no identity-based head selection, was then actually
applied to both full train panels by `scripts/screen_registration_fit_capacity.py`.
Only the four selected midpoints had entered the additional200 updates.
Report: `experiments/registration_fit_capacity_train160_transfer_01/report.json`.

| Unchanged full-panel joint criteria | Initial completed residual | Shared four-failure fit | Gains / losses vs initial |
|---|---:|---:|---:|
| All160 train midpoints | 117/160 (73.125%) | 69/160 (43.125%) | 7 /55 |
| Other156 midpoints, excluding the four fitted observations | 117/156 (75.000%) | 65/156 (41.667%) | 3 /55 |
| All160 train quarter-position observations | 115/160 (71.875%) | 72/160 (45.000%) | 4 /47 |

The midpoint CPU initialization reproduces every saved GPU pass/fail identity.
After this narrow fit, midpoint IoU failures rise from31 to86 and centroid-shift
failures from6 to31; the existing cycle/support/topology failure counts do not
increase. Counts overlap. This is substantial damage to other previously learned
alignments, not a successful repair. **The fitted head is rejected for deployment
and generator training.** All original GPU checkpoints remain unchanged.

These results rule out an inability to fit these four frames with the current
parameterization. They do not prove that the original full-training failures
were caused only by forgetting or that longer training cannot help. They do
justify testing a different fitting procedure: retain broad replay of previously
passing train observations while fitting failures, and compare original-geometry
replay with preservation of the original model's outputs on those observations.
Both arms must start from the original117/160 checkpoint, not this regressed
head, and be assessed on all fixed train panels. No validation/test labels may
enter that fit, and improvement on weak engineering proxies would still not
establish independent physical qualification.

Three new tests cover deterministic failure-only/prefix selection, exclusion of
non-alignment failures and malformed panels, and rejection of unbound diagnostic
specifications. Existing fitting tests reject probe-role observations entering
the optimizer. The independent heads and shared head use a diagnostic-only
checkpoint type and are not accepted as completed full-training pilots.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_fit_capacity \
  --out-dir experiments/registration_fit_capacity_train4_repeat01

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.screen_registration_fit_capacity \
  --fit-root experiments/registration_fit_capacity_train4_repeat01 \
  --out-dir experiments/registration_fit_capacity_train160_transfer_repeat01
```

The full-train comparison entry point is
`scripts/train_registration_residual_pilot.py`, with the bash wrapper below.
This is a **new10-epoch feasibility comparison, not a shortened replacement for
the final300-epoch experiment**. Both arms start from the completed v7 geometry
checkpoint SHA256 `068e3b9a43d5bf559521dce2e7ff6fb7adc406b8f9cbb747783d636f682768bf`.
The tiny-panel diagnostic head is **not** reused or selected as an initialization.

| Fixed setting | Predictor continuation | Residual head |
|---|---|---|
| Base model | Completed v7 geometry | Same |
| Trainable parameters | Existing predictor; eval-mode normalization | New head; frozen base |
| Added epochs / updates | 10 /3,200 | Same |
| Batch / seed | 8 /0 | Same |
| Samples | 16 rotating pairs per160 train sequences each epoch | Same identities/order |
| Objective | Existing v7 geometry, no MIND | Same |
| Initial learning rate | 1e-5 | 1e-4 |
| Scheduler | Cosine to10% of initial LR | Same schedule shape |
| Weight decay | 1e-5 | Same |

Both optimizers start fresh. "Continuation" means additional predictor fine-tuning,
not restoration of v7's exhausted optimizer schedule. The sampler restarts the
same seed0 epoch schedule. Equal updates/data do not make this an isolated
architecture ablation: trainable capacity and learning rates explicitly differ.
No validation/test pixels or annotations are used during this pilot.

First, CPU-only preflight (does not create training outputs or require CUDA):

```bash
cd <repository root>
bash scripts/run_registration_residual_pilot.sh both preflight
```

With two separately allocated GPU terminals, run one command per terminal:

```bash
bash scripts/run_registration_residual_pilot.sh predictor_continuation
```

```bash
bash scripts/run_registration_residual_pilot.sh residual_head
```

Alternatively, sequentially on **one** GPU (do not also launch the commands above):

```bash
bash scripts/run_registration_residual_pilot.sh both
```

The wrapper preserves scheduler `CUDA_VISIBLE_DEVICES`, unsets `LD_LIBRARY_PATH`
only in child commands, and runs CPU regression tests first. Outputs go to
`experiments/registration_residual_pilot_e10_seed0/{predictor_continuation,residual_head}`.
Original v7 runs are not modified. `AERO_RESIDUAL_OUTPUT` selects a fresh root for
a repeat; `AERO_V7_OUTPUT` identifies the completed v7 input root, not new outputs.
Do not run two jobs against the same arm/output. Sources are snapshotted and
checked; do not edit the trainer, head or their recorded dependencies mid-run.

After interruption, resume the affected arm from its last saved epoch boundary:

```bash
bash scripts/run_registration_residual_pilot.sh residual_head resume
```

Substitute `predictor_continuation` for the other arm. Model, optimizer, scheduler
and Torch RNG states are restored; an incomplete epoch is replayed and prior
attempt logs remain. A directory with no valid `latest.pth` cannot be resumed.
An already completed arm is verified without updates/overwrites. CUDA grid-sample
backward may remain nondeterministic; no bitwise GPU replay claim is made.

After **both** arms finish, CPU verification and comparison:

```bash
bash scripts/run_registration_residual_pilot.sh both verify
```

This writes `comparison.json` under the new root. An identical existing report is
verified without overwriting it; a differing report fails closed. Each completed
arm contains a final checkpoint, before/after160-frame train midpoint screens,
completion record, source snapshots and per-batch consumed-input hashes. Verification
checks retained optimizer updates across resume attempts, source/cache identities,
checkpoint binding and matched data across arms. It does not replay training or
rehash every cached array. Completion and even95% train pass still do not grant
registration qualification or generator training.

## Subsequent model-inference audit

`scripts/audit_registration_residual_pilot.py` loads the **new** checkpoint kind
with its architecture-specific loader. Do not pass these `final.pth` files to the
legacy dense matcher or v7 checkpoint loaders. Both arms expose centre-coordinate
fields directly through `model.fields`; the audit applies no second raw-field
conversion. Training scripts and their recorded dependency hashes are unchanged.

The inference audit verifies both completed pilots and their matching data/budget
first. At least one must reach both unchanged 95% train-midpoint criteria before
validation annotations are selected or pixels are read. Once eligible, **both**
models are evaluated on the same predetermined observations; the control is not
omitted simply because it failed its train screen. There is no fitting, threshold
search, checkpoint selection or new frame-lag optimization in this command.

After both GPU pilots finish, CPU-only preflight:

```bash
bash scripts/run_registration_residual_audit.sh preflight
```

Only if it reports `ready: true`, run on one allocated GPU:

```bash
bash scripts/run_registration_residual_audit.sh screen
```

The screen uses eight eligible, evenly selected frames per official train and
validation sequence, as in the existing v4/v7 audit. After reviewing that result,
the `exhaustive` mode evaluates all eligible pairs with the same frozen criteria:

```bash
bash scripts/run_registration_residual_audit.sh exhaustive
```

Do not launch screen/exhaustive now: both completed GPU pilots remain below95%.
The earlier CPU preflight in
`experiments/registration_residual_preflight_audit_01/preflight.json` returned
exit2 due to then-missing pilot files; that historical report does not describe
current completion. The products now verify, but neither final train rate meets
the frozen readiness rule. No held-out pixel inference is authorized by completion.

Audit output directories are fresh-only; the completed preflight above cannot
be overwritten. To record a new preflight, select a fresh output; with the current
results it will still report `ready=false` (now because of the train rates):

```bash
AERO_RESIDUAL_AUDIT_OUTPUT="$PWD/experiments/registration_residual_preflight_audit_02" \
  bash scripts/run_registration_residual_audit.sh preflight
```

The wrapper shares `AERO_RESIDUAL_OUTPUT`, `AERO_REGISTRATION_CACHE` and
`AERO_PYTHON` with training; `AERO_ANTIUAV300_ROOT` selects the native dataset.
It preserves scheduler GPU visibility. Reports bind checkpoint/log/spec bytes,
annotation identities, decoded input hashes, row files and the audit sources.
The report includes paired wins/losses as well as per-arm rates. A failed or
interrupted audit preserves partial products without publishing `report.json`.

Twelve synthetic CPU tests cover readiness, no validation/GPU access on blocked
paths, direct centre-field inference, matched observations, checkpoint-arm/drift
rejection and refusal to overwrite. Separately, both actual zero-epoch checkpoint
types from the earlier CPU smokes were loaded and evaluated on two cached train
observations: both produced finite fields and identical paired outcomes. That
is an interface smoke, **not** a completed residual pilot or a performance result.

Even an exhaustive engineering pass retains generator HOLD. The existing
validation set has been inspected during prior development and is not an untouched
test set. These box/cycle/topology checks do not independently establish physical
RGB-to-IR correspondence, nor do they complete the planned300-epoch experiment.

Before issuing these GPU commands, both arms actually completed two-update CPU
training smokes under `experiments/registration_residual_pilot_cpu_smoke_02/`.
Their consumed frame/byte signatures matched, their initial geometric losses
were identical (0.3423455), and both had finite gradients and second-update losses.
The final source hashes match those smoke snapshots. Both wrapper preflights and
the full523-test CPU suite passed. These are implementation checks; no full GPU
pilot has been run by the assistant and no completion is inferred from preflight.
