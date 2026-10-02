# Uniform versus failure-aware residual fine-tuning

## Completed GPU comparison (2026-09-24)

Both arms completed all **300 additional epochs / 96,000 retained optimizer
updates**. `bash scripts/run_registration_replay.sh both compare` passed the
source-snapshot, checkpoint/optimizer/scheduler, committed-journal and saved-screen
checks. This is artifact verification, not a replay of training or inference.
Uniform resumed on node31 from its completed epoch196 checkpoint:276 interrupted,
uncommitted updates were discarded, not counted twice. No budget was shortened.

| Seed0 train screen | Uniform | Failure-aware |
|---|---:|---:|
| Midpoint joint passes | 125/160 (78.125%) | 124/160 (77.500%) |
| Quarter-frame joint passes | 133/160 (83.125%) | 133/160 (83.125%) |
| Initially failed midpoints repaired | 12/43 | 13/43 |
| Initially passing midpoints regressed | 4/117 | 6/117 |

The midpoint paired comparison has120 both-pass,31 both-fail, five uniform-only
and four failure-aware-only cases. Quarter frames have131 both-pass,25 both-fail
and two unique passes per arm. Thus the failure-aware policy has **no observed
aggregate advantage in this single-seed comparison**. Both miss the unchanged
95% train prerequisite; no validation audit or paired generator is approved.
Quarter observations may have occurred in training and are not a held-out test.

Authoritative comparison:
`experiments/registration_replay_e300_seed0/comparison.json`, SHA256
`7af7352d22a58e2fbd79eb1ca0bea916ad1b1eba2d7cee053228ab6392c90207`.
Final checkpoint SHA256 values:

- Uniform: `f7a057317f8c8348760210c7750230b4ed37f5e689e102426fdad3911b5b647e`.
- Failure-aware: `a45964c3fcf18e6a31dda25e873adb06c57896179cefd8bf604054e2611bdb5d`.

### Remaining failures: measured CPU attribution

The existing constraint analyzer recomputed all midpoint predicates and reread
the exact cached image/box bytes against the saved160 sample hashes per arm.
No model fitting, GPU inference, gate change or new validation access occurred.

| Failed midpoint predicate (overlapping counts) | Uniform | Failure-aware |
|---|---:|---:|
| Any joint predicate | 35 | 36 |
| Box IoU | 26 | 22 |
| Absolute area-ratio change | 21 | 21 |
| Centroid shift | 5 | 4 |
| Global/ROI cycle-error magnitude | 0 | 0 |

Some observations also fail support, bounds or ROI Jacobian checks; zero cycle
magnitude failures does not mean every geometric safety check passed.

| Smaller annotated RGB/IR short side on the cached input grid | Frames | Uniform passes | Failure-aware passes |
|---|---:|---:|---:|
| Below4 pixels | 1 | 0 | 0 |
| 4 to below8 pixels | 20 | 11 | 12 |
| 8 to below16 pixels | 121 | 98 | 96 |
| At least16 pixels | 18 | 16 | 16 |

Small targets have a lower observed pass fraction, but25 of35 uniform failures
are at least8 input pixels across their shorter annotated side. This is not solely
a sub-eight-pixel problem. Fixing only the ten smaller failing observations,
while leaving all others unchanged, would reach135/160, not152/160. This arithmetic
does **not** show that resolution changes cannot help larger objects, or establish
resolution as a causal explanation. Likewise, further reducing already-passing
cycle magnitudes alone would not directly repair the failing alignment predicates.

Reports (diagnostics, not independent correspondence evidence):

- `experiments/registration_replay_uniform_midpoint_constraints_01.json`
  (`4becc7de627c3d1ccb5882abb37162ee3614f301fc0a9c6a5da63d015a11efcf`).
- `experiments/registration_replay_failure_aware_midpoint_constraints_01.json`
  (`ff3353d95c5e6ced93c9edd157dca760d55eb4ab778dd560829c886189900d1e`).

The next repair must test image-conditioned target position/extent estimation
and retention of existing successes, rather than assuming another identical
longer run, failure oversampling or cycle tightening is sufficient. The result
does not establish which replacement method will succeed. Independent physical
correspondence and registered-export authorization remain separate unsolved needs.
The subsequent [box-violation loss ablation](registration_alignment_violation.md)
tests one concrete optimization change on CPU, with matched data and update count;
it is not a new qualification rule or approval to launch another long GPU run.

## Decision and evidence

Registration remains **HOLD**. The completed10-epoch GPU residual model passes
117/160 train midpoints, versus115/160 for predictor continuation and114/160 at
their shared starting checkpoint. The unchanged engineering threshold is95%.
These are box/cycle/support/topology checks, not independent physical pixel GT.

Subsequent CPU diagnostics explain why simply fitting a few failures is not an
acceptable repair. Four selected alignment failures can all be fitted by the
existing shared head and loss, but that narrowly fitted head reduces full-panel
passes to69/160 midpoints and72/160 quarter frames. It is rejected.

The next completed CPU diagnostic starts again from the preserved117/160 GPU
checkpoint. Each of200 updates includes the same four failure midpoints plus
four cyclically sampled passing midpoints, drawn from117 passing sequences.
Both arms use identical samples, initialization, geometry loss, AdamW learning
rate1e-4 and update count. One arm additionally penalizes changes from the initial
model's mapping on passing observations (ROI pixel SmoothL1 plus0.05 global
pixel SmoothL1, auxiliary weight1). The initial mapping is a teacher, not GT.

| CPU diagnostic | Train midpoints | Train quarter frames |
|---|---:|---:|
| Preserved initial GPU residual model | 117/160 | 115/160 |
| Failure fitting with passing-case replay | 118/160 | 116/160 |
| Same replay plus output-preservation loss | 120/160 | 112/160 |

All four targeted failures pass after either fit. Midpoint replay gains five and
loses four; quarter replay gains four and loses three. Output preservation gains
four and loses one midpoint, but gains no quarter observations and loses three.
The extra preservation objective is **not adopted**: its midpoint gain does not
transfer to the other time positions. Replay's net+1 on each panel is weak
evidence, not established improvement or qualification. Quarter frames were not
used in this200-update diagnostic; they remain official train frames from
previously trained sequences, not a held-out test.

Actual saved artifacts:
`experiments/registration_replay_preservation_train4_replay117_01/report.json`.
Both diagnostic checkpoints reload exactly and contain200 AdamW steps. None of
these diagnostic heads initializes the new GPU experiment.

## Frozen GPU comparison

Question: can broad rotating-frame exposure repair failures while retaining
previous successes, and does balanced failure/pass sampling improve on uniform
sequence sampling under the same budget?

- Both arms initialize from the original completed residual GPU `final.pth`,
  SHA-256 `86ba5a3d068645a3275934ed9572bd8502075e32b477b2ed9527d05014e89dcb`.
- Both fine-tune only the residual head for **300 additional epochs**,320 updates
  per epoch, batch8:96,000 updates and768,000 frame draws per arm. These are not
  the earlier10 epochs renamed as300, nor a completed entire research pipeline.
- Both use the unchanged v7 geometry objective, AdamW1e-4, weight decay1e-5,
  gradient clipping5, cosine decay to1e-5. Both start fresh optimizer/scheduler
  states from identical model weights. No MIND/scale/preservation auxiliary.
- `uniform`: each of160 sequences contributes16 frames per epoch.
- `failure_aware`: each batch contains4 sequences from the43 initially failing
  sequences and4 from the117 initially passing sequences. Failure sequences
  contribute29 or30 frames/epoch; passing sequences10 or11. The partition is
  frozen from the original train-midpoint screen, not recomputed during fitting.
- Each sequence has a deterministic shuffled frame stream. Its cursor advances
  by its actual number of draws; all its frames are visited before another
  cycle. Unequal epoch counts do not reset or skip frame-stream positions.
- Model prediction still receives only images and derived geometry, never
  sequence IDs, boxes, partition membership or fitted per-frame targets.
- This matches initialization/model/loss/update budget, **not consumed training
  data or batch composition**. Attribute any difference to the sampling policy,
  not to isolated architecture changes or identical-data training.
- No validation/test access, best-epoch selection or early-stop tuning. The final
  checkpoint is evaluated on all160 train midpoints and all160 quarter frames,
  including paired gains/losses. These frames can occur during training.
- Passing these train screens alone still does not establish physical
  correspondence, authorize generator training or complete qualification.

## Commands

Run from the repository root with the existing virtual environment. No pip
installation, data download or original checkpoint overwrite is required.
On **two independently allocated GPUs**, execute one command per allocation:

```bash
# GPU allocation A: uniform control
bash scripts/run_registration_replay.sh uniform train
```

```bash
# GPU allocation B: failure-aware candidate
bash scripts/run_registration_replay.sh failure_aware train
```

The wrapper preserves scheduler GPU visibility; it does not hard-code GPU0.
With only one allocated GPU, `bash scripts/run_registration_replay.sh both train`
runs the two arms sequentially and compares after both finish.

For a walltime kill, obtain a new allocation and resume only the interrupted arm:

```bash
bash scripts/run_registration_replay.sh uniform resume
bash scripts/run_registration_replay.sh failure_aware resume
```

Each command resumes its own last completed epoch. The optimizer, scheduler and
RNG states are restored. Uncommitted updates from a killed epoch are explicitly
discarded in the lineage audit and rerun; they are not counted as completed.
Do not launch a resume concurrently with an existing process. A kernel lock
also rejects two writers to the same arm. Do not edit recorded source files,
change the environment or delete outputs during training.

After both finish, compare on CPU:

```bash
bash scripts/run_registration_replay.sh both compare
```

Outputs: `experiments/registration_replay_e300_seed0/{uniform,failure_aware}/`
and `experiments/registration_replay_e300_seed0/comparison.json`.
An existing completed run is verified without retraining by `resume` or `verify`.
`train` never overwrites an existing arm. To intentionally start a separate
experiment, set a fresh single-line `AERO_REPLAY_OUTPUT` consistently for both arms.

CPU-only implementation checks (not scientific evidence):

```bash
bash scripts/run_registration_replay.sh both preflight
bash scripts/run_registration_replay.sh both smoke
```

Both real-data CPU smokes completed under
`experiments/registration_replay_e300_seed0_cpu_smoke/`: two finite-gradient
optimizer updates per arm, exact checkpoint tensor reload and committed-journal
verification. The saved checkpoints also completed the midpoint/quarter evaluator
on two real train sequences per arm; this tests the interface, not performance.
All609 CPU tests, repository Ruff checks and `git diff --check` passed. Source
files still matched both smoke snapshots, and the original completed GPU pilot
comparison reverified without changes. No new300-epoch GPU run has been executed
by the assistant.

`smoke` uses a separate `_cpu_smoke` output directory and only two updates per
arm; it cannot verify or compare as a300-epoch result. Tests cover deterministic
sampling, uninterrupted per-sequence frame coverage, partition rejection,
checkpoint reload/optimizer continuation, interrupted-journal rollback and smoke
rejection. The saved-artifact verifier reconstructs selected frames and binds the
committed log chain to the checkpoint; it does not replay all GPU computations
or freshly rehash every cached array. CUDA nondeterministic operations are warned,
not claimed to be bitwise reproducible across devices.

## Native-input engineering audit after training

`scripts/audit_registration_replay.py` loads the new checkpoint type directly.
The older residual-pilot auditor accepts the10-epoch checkpoint type and must
not be used to reinterpret these300-additional-epoch checkpoints.

First finish both arms and their comparison. Then run the CPU-only readiness
check; it does not launch another training run:

```bash
bash scripts/run_registration_replay_audit.sh preflight
```

Readiness requires both complete, verified300-epoch products, and at least one
arm meeting both unchanged95% train-midpoint rates (joint-frame and sequence
macro). Quarter-frame results remain reported diagnostics, not an additional
post-hoc selection rule. If products are missing or inconsistent, or neither arm
meets those rates, the auditor writes `preflight.json`, returns exit2 and does
not read validation annotations/videos or initialize CUDA. Missing artifacts do
not establish that a remote process stopped; process state is not inferred.

Only after a ready preflight, use an allocated GPU for the eight-frame screen:

```bash
bash scripts/run_registration_replay_audit.sh screen
```

The final eligible-pair coverage audit has a separate output directory:

```bash
bash scripts/run_registration_replay_audit.sh exhaustive
```

These are inference-only commands, not further optimization. They compare both
models on exactly the same annotation-selected train/validation frame pairs,
without changing the frozen same-index pairing or thresholds. They use direct
centre-coordinate forward/reverse fields, not the old raw-flow adapter. Row
files bind decoded input bytes, frame identity and both-direction measurements;
coverage checks reject missing/duplicate frames. Source files, input annotation
identities and training-product hashes are recorded and rechecked. Interrupted
or inconsistent evaluation never publishes a final `report.json`.

Outputs default to `experiments/registration_replay_{MODE}_audit_01/`, where MODE
is `preflight`, `screen` or `exhaustive`. Directories are fresh-only; after an
interruption or a previously blocked preflight choose a new single-line
`AERO_REPLAY_AUDIT_OUTPUT`. Do not delete or overwrite the earlier evidence.
The wrapper shares `AERO_REPLAY_OUTPUT`, `AERO_RESIDUAL_OUTPUT`,
`AERO_REGISTRATION_CACHE`, `AERO_ANTIUAV300_ROOT` and `AERO_PYTHON` with the
training configuration where applicable.

Even an exhaustive engineering pass retains **generator HOLD**: the existing
validation set has been inspected during earlier development, and box/cycle/
topology measurements do not establish independent physical pixel
correspondence. This auditor is not a scientific qualification/export authorizer
and its report must not be relabelled as a supported generator-training pass.
The benchmark paper also describes train/validation as non-overlapping clips
from shared original videos: sequence-ID separation alone is not independent
recording separation. See the [dataset evidence audit](registration_dataset_evidence.md).

Sixteen synthetic audit tests cover frozen readiness thresholds, no validation/
CUDA access when blocked, no decoding during ready preflight, direct-coordinate
inference, paired input identity, missing/duplicate coverage, artifact drift,
source snapshots and refusal to overwrite. Mocked training verifiers in these
wiring tests are explicitly marked; their perfect identity-model scores are
not research results. A real CPU preflight also observed missing300-epoch
products in the default output location and correctly stopped before evaluation.
That historical preflight does not describe a future GPU run's status.
The final-source check is saved at
`experiments/registration_replay_preflight_cpu_check_02/preflight.json`.
With the auditor added, all625 CPU tests, repository Ruff checks and shell syntax
checks passed. The existing training sources still match both completed CPU-smoke
snapshots; no training-protocol edit or GPU restart is required by this addition.
