# Temporal-exposure box-violation diagnostic

This is a bounded CPU experiment, not registration qualification or a substitute
for the completed300-epoch GPU comparison.

## Motivation and design

The preceding [midpoint-only loss diagnostic](registration_alignment_violation.md)
raised fitted passes from125 to136, but quarter-frame passes fell from133 to132.
This motivates testing whether temporal exposure changes that transfer result,
not assuming that the auxiliary already repairs the registration problem.

The executable protocol is
`configs/experiment/registration_temporal_violation_cpu.yaml`.
Both arms again start from the **original completed uniform300 checkpoint**, not
either midpoint-fitted diagnostic head. The frozen base predictor is unchanged.

For each of the160 official train sequences:

1. Exclude the exact cached midpoint and quarter-frame indices.
2. Select ten distinct, evenly spaced positions from the remaining ordered cache
   indices, using only the sequence length, never image scores or annotations.
3. In each of ten time rounds, include every sequence once in seeded shuffled
   batches of eight, without duplicate sequences within a batch.

Each arm therefore consumes1,600 distinct frame pairs in200 updates. This matches
the preceding diagnostic's frame-draw/update budget and optimizer settings, while
changing which frames are consumed. Comparisons between these two diagnostics are
not identical-data comparisons. The control and auxiliary arms **within** the
present diagnostic do consume identical frames in identical batches.

The control retains the v7 geometric loss. The candidate adds the same worst
per-pair box-violation term with weight1. Both use AdamW1e-5, weight decay1e-5,
clipping5 and seed0; only the residual head is trained. Existing registration
thresholds and all original safety losses remain unchanged. Boxes enter the loss
and evaluation only, not prediction. There is no early stopping or best-epoch
selection.

Both full160-frame midpoint and quarter panels are evaluated before and after
training. These320 frames are excluded from the present updates, **but may have
been consumed in the earlier300-epoch training**. They are temporal-transfer
diagnostics on old training sequences, not validation/test evidence.

The runner verifies the completed comparison first and requires CPU initial
pass/fail decisions to reproduce the saved GPU decisions. It records per-batch
input hashes shared by both arms, rechecks consumed training/probe bytes, and
saves/reloads both diagnostic heads before final evaluation. Original checkpoints
and source snapshots are retained unchanged. No outcome grants generator approval.

## Executed result (2026-09-24)

The CPU run completed in922.03 seconds. Both arms completed200 optimizer updates
on the same1,600 distinct training pairs. The consumed training-pair identities
are disjoint from the320 midpoint/quarter probe identities; input hashes match
between arms on all200 batches. Initial CPU decisions reproduce the saved GPU
decisions on both panels. Both final heads reload exactly, and original inputs,
source snapshots, trace and checkpoint hashes verify. All664 CPU tests and
repository lint checks pass.
An additional saved-state check finds changes in all18 head tensors in both
arms (parameter-delta L2 norms0.0196144 and0.0209546, respectively). Unchanged
aggregate pass counts are therefore not caused by a no-op parameter update.

| Condition | Midpoint passes | Quarter passes | Midpoint gains / losses | Quarter gains / losses |
|---|---:|---:|---:|---:|
| Completed uniform300 initializer | 125/160 (78.125%) | 133/160 (83.125%) | — | — |
| Temporal frames, original geometry loss | 125/160 (78.125%) | 133/160 (83.125%) | 0 / 0 | 1 / 1 |
| Temporal frames, geometry plus violation | 125/160 (78.125%) | 133/160 (83.125%) | 0 / 0 | 1 / 1 |

Gains/losses are relative to the initializer. Neither condition improves aggregate
pass counts. Broader temporal exposure under this budget does not carry the
preceding midpoint-fit gain into these probe panels. This does not prove every
temporal sampling scheme, auxiliary weight or longer budget ineffective; it does
reject an observed improvement claim for the implemented diagnostic.

**Decision:** neither diagnostic head is promoted and no new long GPU run is
justified solely by this result. The remaining image-conditioned alignment and
independent physical-evidence problems are unresolved. Qualification and generator
training remain HOLD. These are negative train diagnostics, not held-out physical
accuracy measurements or results of a completed end-to-end pipeline.

Actual report: `experiments/registration_temporal_violation_cpu_01/report.json`,
SHA256 `691b792eb2f11ad14bb34aec50e8fdb8a332cc8a8d0b2a80fbc1bb7f4ff14b68`.
Saved head SHA256 values:

- Geometry only: `49def60be8a499916e83ebcae9be8ba4e224329fda8d6390d431c378910863ab`.
- Geometry plus violation: `80bdc43d6a4711a0e5b698181c47d41b7466103ded4346ecb960ee2087f1da32`.

## Reproduction

Use a fresh output directory:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/aero-registration-mpl \
  .venv/bin/python -u -m scripts.probe_registration_temporal_violation \
  --config configs/experiment/registration_temporal_violation_cpu.yaml \
  --out-dir experiments/registration_temporal_violation_cpu_repeat01
```

The presence of this command, a source file or a passing test is not a result.
Use a completed `report.json` for numerical conclusions; no GPU run is launched
by this CPU-only runner.
