# Shared-velocity learning pilot (v7): protocol and evidence checks

## Status and research boundary

The image-conditioned shared-velocity training path is implemented. CPU tests,
real-data optimizer smoke checks for both arms, and both GPU-command preflights
have completed. Both arms have completed10 epochs and3,200 updates each; their
saved products pass the evidence verifier, including matched consumed-data
signatures. Train midpoint joint pass is114/160 (71.25%) for `geometry` and
115/160 (71.875%) for `geometry_mind`, both below95%. Neither is a candidate for
the next held-out engineering screen under the frozen rule.
The user executes GPU commands on an allocated node. No GPU training was started
by the assistant; its CUDA tensor check failed and the escalation was cancelled.

This is a new10-epoch feasibility comparison, **not a reduced replacement for the
planned300-epoch final experiment**. Neither a completed pilot nor a95% train
geometry screen clears held-out/exhaustive registration or paired-generator gates.
No original checkpoint, pairing, validation/test data or qualification threshold
has been changed. The project is not yet ready for a completed-results release.

## What is trained

`protocol_v7.SharedVelocityMatcher` makes one image-conditioned prediction using
the fixed v6 architecture, interprets its centre-coordinate output as a stationary
velocity, and jointly computes exp(v) and exp(-v) with seven integration steps.
Both maps participate in the training objective and update the same predictor.
The raw/centre coordinate adapter is applied explicitly for legacy audit callers.

Naive initialization is known to reduce box accuracy; this is measured in the
[initialization diagnostic](registration_shared_velocity.md), not hidden as a
preserved map. V7 now optimizes native alignment as well as reciprocal constraints.
The geometry objective retains box IoU/centroid/area, smoothness, cycle tails,
Jacobian and support terms; the old full-image edge-NCC term is excluded.

| Frozen setting | Geometry control | Geometry + MIND |
|---|---|---|
| Initial checkpoint | completed v6 seed0, SHA256 `b719c61b…cf6f02e` | same |
| Epochs / batch / seed | 10 /8 /0 | same |
| Official train sequences | 160 | same |
| Pairs per sequence per epoch | 16 rotating positions | same |
| Optimizer updates | 320 per epoch,3,200 total | same |
| Learning rate | 1e-5, cosine to1e-6 | same |
| Shared transform / geometric objective | enabled | same |
| MIND descriptor loss weight | 0 | 0.05 |

The MIND arm uses the existing explicitly defined2D descriptor. Existing train boxes
define1.5x target observation regions; a fixed texture threshold1e-6, three-pixel
descriptor margin, header/thin-reticle heuristic and8px guard exclude suspect
pixels. This is weak box-supervised training, not annotation-free learning.
Heuristic HUD exclusion can remove real content or miss overlays.

The observation denominator is fixed by the target image. Unobserved warped source
pixels incur descriptor loss1 rather than disappearing from the average. Pairs
without target texture contribute no MIND evidence and are reported separately;
their absence cannot be interpreted as successful matching. Perturbation masks and
SAM outputs are not used as fine-grained GT. MIND remains a candidate loss, not an
independent accuracy measurement.

## Completed matched-pilot result

Saved-artifact verification was rerun in
`experiments/registration_v7_evidence_completion_review_01.json`. This checks
checkpoint/source identities, completion records, optimizer/scheduler budgets and
all recorded training selections; it is not a GPU inference or training replay.

| Train midpoint screen (one frame per sequence) | Joint pass | Sequence macro |
|---|---:|---:|
| Shared-velocity initialization | 68/160 (42.50%) | 42.50% |
| Geometry,10 epochs | 114/160 (71.25%) | 71.25% |
| Geometry + MIND,10 epochs | 115/160 (71.875%) | 71.875% |
| Frozen candidate threshold | 152/160 (95%) | 95% |

MIND adds one passing observation (`20190925_143900_1_4`) and loses none relative
to the matched geometry control. Relative to initialization, the control gains49
and loses3 observations; MIND gains50 and loses3. This single-seed, train-only
comparison does not establish a statistically reliable or held-out MIND benefit.

In the MIND final screen,45 frames fail at least one criterion. Overlapping
failures include34 IoU,28 area-ratio and6 centroid cases; cycle-error magnitude
criteria fail on zero frames. Support and local Jacobian failures remain. At
least37 additional joint passes are needed just for the train midpoint threshold.
See `experiments/registration_v7_mind_final_scale_01.json`; no criterion was relaxed.

A separate CPU diagnostic at the **completed geometry control** measures gradients
with respect to its detached shared velocity, not network parameters. Across160
train midpoint observations, the weighted MIND/geometry gradient norm ratio has
median0.005354 (~0.535%). Gradient cosine has median0.01476, with85 positive,
72 negative and3 undefined (zero MIND gradient). The MIND training logs also show
98.55% mean target eligibility, so the term is not simply disabled. These findings
motivate testing the alignment update mechanism; they do not justify blindly
increasing MIND weight or establish the cause of Adam/network behavior.
Artifact: `experiments/registration_v7_geometry_mind_gradient_01.json`.

### Annotation-assisted residual feasibility (not performance)

`scripts/probe_registration_v7_refinement.py` tests whether a small smooth residual
velocity can improve completed-control alignment under the unchanged geometry
objective. It freezes the predictor and fits an independent16x16 control grid
on each of the first four lexicographic saved failures and four saved passes.
Defaults are100 Adam updates, learning rate0.005,24-network-pixel bounded residual
velocity, boundary taper and the same seven-step exp(+v)/exp(-v) construction.
Only the final iterate is scored; no best-gate iterate is selected.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_v7_refinement \
  --out experiments/registration_v7_refinement_train8_01.json
```

This intentionally fits and scores the same existing **train boxes**. It is an
oracle feasibility diagnostic, not independent correspondence evidence, a learned
image-conditioned correction, or a dataset pass-rate estimate. No corrected maps
are exported for generator use. Success would motivate an image-conditioned
residual predictor; failure would not prove that representation is impossible.

The default CPU experiment actually completed in60.1 seconds. Three of the four
selected failures became joint passes; all four selected passes remained passes.
For failure IDs ending `101846_1_3`, `101846_1_6`, and `101846_1_7`, the worse of
the two directional box IoUs changed from0.564/0.449/0.200 to0.992/0.992/0.935,
respectively, while satisfying all the unchanged predicates. The fourth failure
(`101846_1_1`) retained invalid ROI support, unobserved Jacobian stencils and an
out-of-bounds mapped box, despite improved IoU. This separates an alignment-fit
opportunity from support limitations on these selected observations only.

Artifact: `experiments/registration_v7_refinement_train8_01.json`. This is **not**
a trained-model improvement from71.25% to87.5%: the panel is deliberately
stratified and annotations were used during per-frame optimization. A practical
repair must predict corrections from images without consulting evaluation boxes,
then pass the original train and held-out checks. No new GPU run is authorized
by this diagnostic alone.

The two-arm CPU audit preflight also completed in
`experiments/registration_v7_completion_preflight_01/preflight.json`: `ready=false`,
exit2 because neither train screen reached95%. No train/validation pixel inference
was launched by that preflight; the annotation-selected panel is recorded separately.

## Launch on allocated GPUs

One GPU, sequential arms:

```bash
cd /lustre/winston1214/project/aero
source .venv/bin/activate
bash scripts/run_antiuav300_registration_v7.sh both
```

Two independently allocated GPU terminals, one command per terminal instead:

```bash
bash scripts/run_antiuav300_registration_v7.sh geometry
```

```bash
bash scripts/run_antiuav300_registration_v7.sh geometry_mind
```

Do not run sequential and parallel commands against the same output simultaneously.
The wrapper preserves scheduler GPU visibility; it never assigns device0 explicitly.
Both outputs default to `experiments/antiuav300_registration_v7_pilot_e10_seed0/`,
with one subdirectory per arm. Set a fresh `AERO_V7_OUTPUT` for a repeat.

CPU-only input/protocol preflight (no experiment directory is created):

```bash
bash scripts/run_antiuav300_registration_v7.sh both preflight
```

Explicit epoch-boundary resume:

```bash
bash scripts/run_antiuav300_registration_v7.sh both resume
```

Resume restores optimizer, scheduler and Torch random state. A partially completed
epoch is repeated from its last saved boundary; attempt logs are preserved. A
directory without an epoch-boundary `latest.pth` cannot be resumed automatically.
Use a fresh output rather than deleting evidence. Do not edit the snapshotted
training sources while running: protocol/source drift fails closed. Documentation
and the independent verifier can be edited without changing the training inputs.

Grid-sample backward may be nondeterministic on CUDA despite seeded inputs and
deterministic-algorithm warnings. This pilot does not claim bitwise training replay.

## CPU evidence already executed

- `experiments/registration_v7_geometry_cpu_smoke_02/`: initial one-frame screen,
  followed by two actual train optimizer updates with batch1.
- `experiments/registration_v7_mind_cpu_smoke_02/`: identical selected frames and
  initialization, with MIND added. The first batch had finite MIND loss0.33516,
  target eligibility1.0 and mean source observation fraction0.81711.
- Both smoke runs report `checkpoint_is_trained_pilot=false`; they are not pilot
  completions or performance comparisons. Earlier`_01` is retained as initial smoke.
- Tests include shared-transform gradients, source-support loss, known translation
  learning, checkpoint roundtrip, legacy-loader rejection, sampler identity and
  consumed-data hashing. Initial evaluation uses no-grad rather than inference mode
  so cached predictor tensors remain valid for subsequent backward passes.

## Verify completed products (CPU only)

```bash
.venv/bin/python -m scripts.verify_registration_v7 \
  --out experiments/registration_v7_evidence_audit_01.json
```

This verifier is separate from the trainer and does not modify its source or outputs.
An existing audit output is never overwritten. Exit0 means both saved pilot products
and their matched comparison passed the evidence checks; **it never means scientific
registration qualification**. Exit2 means missing/incomplete/invalid/unmatched products.
Missing files do not imply a running job, a stopped job, or training success.

Checks include:

1. Frozen10-epoch protocol, no smoke substitution, original source snapshots and
   compatible current verifier dependencies.
2. Full160-sequence cache identity and hashes, exact rotating sampler choices,
   finite per-batch training measurements and complete update budgets.
3. Epoch-boundary resume lineage: discard abandoned attempt tails before counting
   updates retained by the final checkpoint.
4. Final checkpoint architecture, loadable finite tensors, specification hash,
   completed epochs, optimizer/scheduler step counts and completion record.
5. All160 unique midpoint observations in both initial/final screens, identical
   sample hashes, and gate recomputation from per-frame bidirectional measurements.
6. Same budget, initialization, source, training order, consumed-byte signatures and
   screen observations between arms; report newly passing and newly failing sequences.

It does **not** re-run inference, re-read every cached training array, establish an
independent initial-checkpoint replay, or supply physical correspondence GT. Those
limits are explicit in the audit JSON. The pending-status smoke report is
`experiments/registration_v7_evidence_audit_pending_01.json`, not a completed result.

## Next decision, without weakening the original objective

### Evaluate completed models on the matched train/validation panel

The legacy v4 audit loader requires a `DM` payload and a completed 300-epoch run;
it must not be used to reinterpret a v7 velocity checkpoint. The new
`scripts/audit_antiuav300_registration_v7.py` loads the verified shared-velocity
architecture and consumes `model.fields()` centre-lattice fields directly. There
is no second raw-to-centre conversion. Both directions are scored on the same
frame using the unchanged v4 thresholds.

**After both v7 training arms finish**, on an allocated GPU:

```bash
bash scripts/run_antiuav300_registration_v7_audit.sh screen
```

This is evaluation, not another training run. It checks both saved pilots using
the evidence verifier above before touching validation pixels. At least one arm
must meet the existing train-screen joint and sequence-macro 95% thresholds to
proceed; otherwise it saves the blocking reason and stops without GPU inference.
When eligible, both arms are evaluated, not just the selected candidate.

The eight-frame-per-sequence annotation-selected panel contains **1,280 train
pairs (160 sequences) and 536 validation pairs (67 sequences)** in the current
dataset. Selection SHA256 is
`4a4d2ce5ed47ea477d752f3eedcc2bfe157b2a10e2794dd322c310a63cde8c48`.
It uses the pre-existing uniformly spaced eligible-frame rule; no scores are used
to select frames or adjust pairing. The two arms consume the same decoded inputs,
whose pixel/box hashes are recorded. Train/validation sequence overlap, duplicate
or missing observations, input/checkpoint/code drift fail the audit. Test data
is not accessed. Validation has been inspected in earlier development diagnostics;
this is **not** a claim of an untouched final test.

The output directory defaults to `experiments/registration_v7_screen_audit_01/`:

- `panel.json`: exact selected frame identities and annotation hashes;
- `preflight.json`: verified pilot evidence, settings and launch eligibility;
- `geometry.jsonl`, `geometry_mind.jsonl`: streamed per-frame bidirectional metrics;
- `report.json`: summaries, matched pass/regression counts and separate gates.

`report.json` is written only after complete inference and post-run integrity
checks. Partial row files or a preflight file do not establish completion. Exit0
for inference means the report was written, **not** qualification. The report's
generator gate remains HOLD because this audit supplies engineering proxies, not
independent physical correspondence evidence. No masks/boxes/cycles are relabelled
as physical GT.

CPU-only preflight can be run using `preflight` in place of `screen`. A missing
pilot or failed train screen returns exit2 with an explicit reason. Existing
audit directories are never overwritten; set a fresh `AERO_V7_AUDIT_OUTPUT` if
repeating an audit. `AERO_V7_OUTPUT` identifies the same training root as the
training wrapper. The evaluation wrapper preserves scheduler GPU visibility.

An `exhaustive` mode is also implemented for all eligible train/validation pairs,
but should follow review of the screen, not be launched to bypass the failed
train-candidate gate. Rows stream to disk to limit memory. Even exhaustive eligible-pair
coverage excludes absent/invalid-box frames and cannot establish dense physical
accuracy or replace the planned 300-epoch final experiment.

Tests use analytic centre-field translations and synthetic end-to-end fixtures;
they cover direction naming, double-conversion regressions, exact coverage,
nonfinite failures, decoded-byte identities, split integrity, and missing-pilot
guards. A CPU integration check on train sequence `20190925_101846_1_1`, frame0,
also ran through the real image-conditioned initializer. This was the **untrained
v7 warm start**, not a completed v7 checkpoint, and has no qualification claim.

### Decision boundary

Review both arms, including coverage and regressions, not just their mean loss.
Meeting the95% train geometric threshold only identifies a candidate for a frozen
held-out engineering screen. A MIND advantage on these training observations is
not yet a generalization or physical-accuracy result. No automatic launch of300
epochs, generator training, or release occurs from these files.

### Explain failures without modifying the gate (CPU)

`scripts/analyze_registration_constraints.py` decomposes the exact existing
per-direction predicates and checks agreement with `direction_pass`. It counts
failure **unions on the same frame**, not the sum of directional failures, and
keeps sequence-macro and frame-micro rates distinct. The input hash and gate-code
hash are recorded. It accepts saved train-screen JSON or audit JSONL rows; reading
a partial JSONL does not establish completed coverage or qualification.

After a final train screen exists, for example:

```bash
.venv/bin/python -m scripts.analyze_registration_constraints \
  --input experiments/antiuav300_registration_v7_pilot_e10_seed0/geometry/final_train_screen.json \
  --out experiments/registration_v7_geometry_final_constraints_01.json
```

Use `geometry_mind` and a distinct output filename for the other arm. This command
does not use a GPU, rerun inference, select a checkpoint or change a threshold.
Existing output files are not overwritten. The product verifier remains necessary.

The analysis was actually executed on the **initial, pre-training** geometry-arm
screen, input SHA256
`fdbe018beb9132a97f596e04534f6261381e0171375ecfca96d14e44d59a6170`.
Artifact: `experiments/registration_v7_initial_constraint_analysis_01.json`.
It is not a v7 trained-model result:

| Initial-screen measurement | Frames out of 160 |
|---|---:|
| All criteria satisfied in both directions | 68 |
| At least one criterion failed | 92 |
| IoU failed in at least one direction | 82 |
| Absolute area-ratio change failed | 46 |
| Centroid shift failed | 30 |
| ROI observed support failed | 2 |
| ROI positive-Jacobian fraction failed | 2 |
| Mapped box bounds failed | 1 |
| Global or ROI cycle-error criteria failed | 0 |

Failure rows overlap. The last row does **not** establish physical correspondence:
a reciprocal transform can still map the wrong object location. Positive-Jacobian
failures can also include unobserved derivative stencils, not just proven folds.
This initializer needs alignment improvement, not merely smaller cycle errors.

As a diagnostic hypothetical, if all IoU/centroid/area failures vanished while every
other recorded quantity remained fixed, joint pass would be 158/160 (98.75%).
Fixing only the cycle group would leave 68/160 (42.5%). These are **not achieved
results or predictions of training**, and constraints may interact in a real model.
At least 84 currently failing frames would need to pass to reach 152/160 (95%) on
this one-frame-per-sequence training screen. Held-out/exhaustive and independent
physical evidence remain separate requirements; no generator approval follows.

### Inspect completed-arm images (CPU)

Once an individual arm has completed and passes the saved-product verifier, its
initial/final train-screen observations can be rendered without waiting for the
other arm or using a GPU:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.render_registration_v7 \
  --arm geometry --out-dir experiments/registration_v7_geometry_visuals_01
```

This is separate from the two-arm GPU audit, which still requires both completed
pilots. Replace the arm and output name for `geometry_mind`. Rendering refuses an
unfinished pilot or an existing output directory; do not use its intermediate
`latest.pth` as a final model. The renderer reads only the already selected train
observations and checks their pixel/box hashes against the saved screen.

It selects the first two sorted sequence IDs in each recorded joint-pass stratum:
newly passing, regressed, both failing, both passing. Missing strata remain empty;
successes are not fabricated. These deliberately stratified diagnostics are not a
representative sample or an estimate of overall performance. Each panel shows
visible input, IR observation, initial RGB warp and trained RGB warp, plus target
zooms. The reference is the **shared-velocity initialization**, not the native v6
model or a separately trained baseline. Cyan/orange boxes are the dataset's RGB/IR
annotations, not fine-grained correspondence GT. The RGB crop has its own native
box; the IR and both warp crops share the IR coordinate window.

Image plotting explicitly uses boundary coordinates `[0,W] x [0,H]` to avoid a
half-pixel mismatch with the annotation boxes. No contrast fitting or synthetic
replacement of the observations occurs. The saved report contains per-image
hashes, selected cache indices, input signatures and freshly recomputed CPU metrics
in both directions. It does not assert bitwise agreement with GPU measurements.
Black warp padding remains unobserved, not newly generated content. Outputs stay
under git-ignored `experiments/`; publication permissions need separate review.

Still required for the requested end state: qualified registration with independent
evidence on the actual supervision domain; actual generator training; the matched
real-only / real+generated / real+synthetic detector comparison; reproducible result
tables and honest documentation; and a reviewed GitHub commit/push under the intended
account. The existing generator preparation script is orchestration, not a completed
diffusion trainer. Human review is not imposed as a prerequisite to testing repairs.

### Completed geometry-control result

The first10-epoch arm is complete and verified, not merely an intermediate loss
log. Final checkpoint SHA256:
`068e3b9a43d5bf559521dce2e7ff6fb7adc406b8f9cbb747783d636f682768bf`.
Final train-screen byte SHA256:
`091a288d493e1307b2cb065433fd4c05ed60ed53fd1316aba660c6197833b345`.
`experiments/registration_v7_evidence_geometry_complete_01.json` records the
earlier individual geometry verification; the completed two-arm comparison is
now recorded in `experiments/registration_v7_evidence_completion_review_01.json`.

| Same160 train midpoint observations | Initialization | Geometry,10 epochs |
|---|---:|---:|
| Joint bidirectional pass | 68/160 (42.5%) | 114/160 (71.25%) |
| Sequence-macro pass | 42.5% | 71.25% |
| Frames failing IoU in either direction | 82 | 35 |
| Frames failing area-ratio change | 46 | 29 |
| Frames failing centroid shift | 30 | 6 |
| Frames failing any cycle-error magnitude bound | 0 | 0 |

Failures overlap; an individual metric improvement is not joint qualification.
There are49 newly passing sequences,3 regressions,43 persistent failures and65
persistent passes. At least38 of the46 remaining failing frames must become joint
passes to reach152/160 on this screen. This does not predict the result of longer
training or establish held-out performance. The trained reference used in this
table is **not** the MIND arm; its matched comparison is reported above.

Actual CPU inference rendered eight stratified examples under
`experiments/registration_v7_geometry_visuals_01/` with per-example recomputed
bidirectional measurements and hashes in `report.json`. Both improvements and
regressions are retained. The failure-attribution report is
`experiments/registration_v7_geometry_final_constraints_01.json`.
These results support an improvement in training-set geometric proxies, **not**
independent physical registration or authorization for paired-generator training.
