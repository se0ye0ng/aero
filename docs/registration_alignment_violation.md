# Box-threshold violation loss: controlled CPU diagnostic

Status: **not registration qualification**. This tests an optimization hypothesis,
not a replacement for independent physical correspondence evidence.

## Question

After300 additional epochs, uniform and failure-aware sampling pass125/160 and
124/160 train midpoints. The remaining failures chiefly concern box overlap and
area. Does penalizing the failing box predicates directly help without sacrificing
existing successes, compared with continuing the exact same geometric loss?

This is not an established root cause: a mean objective need not maximize a
thresholded pass count, but that observation does not prove a hinge penalty will
repair these images. The ablation tests that hypothesis instead of launching
another full GPU run on it.

## Frozen diagnostic method

The YAML experiment is
`configs/experiment/registration_alignment_violation_cpu.yaml`.
Both arms begin with the completed uniform300 checkpoint, whose source, optimizer
budget, journal and screen identities are verified before fitting. Only the
residual head is trainable; the base predictor and its image-derived context are
frozen. Ground-truth training boxes enter losses/metrics, never prediction.

For each pair and each direction, the auxiliary computes three nonnegative,
dimensionless violations using the **existing** qualification constants:

- `max(0, 1 - IoU / 0.6)`;
- `max(0, centroid_fraction / 0.25 - 1)`;
- `max(0, absolute_area_ratio_change / 0.5 - 1)`.

The largest of the six component/direction violations is taken within each pair,
then averaged over every pair in the batch. It does not average a failed direction
away against an easy direction or drop passing observations from the denominator.
The candidate adds this term with weight1 to the existing geometry loss; the
control uses weight0. Existing geometry terms, including alignment, support,
topology and cycle constraints, remain active. A zero auxiliary alone is never
treated as a registration pass.

Both arms use seed0, batch8,200 AdamW updates, learning rate1e-5, weight decay1e-5
and gradient clipping5. Each of the160 train midpoint frames is used exactly ten
times, with the same shuffled batch order in both arms. There is no checkpoint
selection or early stopping. This diagnostic is **not** a shortened substitute
for the completed300-epoch training or a new full-training experiment.

CPU inference must reproduce the original GPU pass/fail decisions on both panels
before fitting begins. Final fitted heads are saved, reloaded with exact tensor
equality, then evaluated on all160 midpoints and all160 quarter frames. Each
panel's cached image/box bytes are checked against the completed run's sample
hashes both before and after the diagnostic. Original files are not overwritten.

The quarter frames are excluded only from these200 extra updates: both panels
belong to sequences already used in the300-epoch run, and the frames may have
been consumed there. They are a transfer diagnostic, not a held-out test.

## Executed result (2026-09-24)

The CPU diagnostic completed in373.33 seconds. Before fitting, CPU inference
reproduced all saved GPU pass/fail decisions on both160-frame panels. Both heads
completed exactly200 optimizer updates, reloaded with exact tensor equality, and
were evaluated from the reloaded tensors. Source snapshots, original inputs,
trace and saved checkpoint hashes passed the post-run checks. All658 CPU tests
and repository Ruff checks passed; original training sources were not modified.

| Condition | Fit midpoints | Quarter-frame probe | Midpoint gains / losses | Quarter gains / losses |
|---|---:|---:|---:|---:|
| Completed uniform300 initializer | 125/160 (78.125%) | 133/160 (83.125%) | — | — |
| Continue original geometry loss | 131/160 (81.875%) | 131/160 (81.875%) | 6 / 0 | 0 / 2 |
| Geometry plus box-violation loss | 136/160 (85.000%) | 132/160 (82.500%) | 11 / 0 | 0 / 1 |

Gains/losses are paired against the initializer. Under the matched update budget,
the new term improves the **fitted panel** by five more observations than the
control, with no lost midpoint passes. It does not improve the quarter panel
against initialization. The probe decline is smaller than the control's, but
zero gains there is not evidence of improved transfer. Neither panel clears95%.
This result supports fit responsiveness to the auxiliary, not its reliability on
other time positions, independent physical accuracy, or overall qualification.

**Decision:** do not promote either diagnostic head, launch another300-epoch GPU
run solely on this result, or start generator training. The hypothesis that a
box-threshold loss alone solves the current registration problem is unsupported
by this bounded test. Any subsequent test must assess broader temporal exposure
and preservation beyond the fitted midpoints; it must keep the existing gates
and the separate physical-evidence requirement.

The [temporal-exposure diagnostic](registration_temporal_violation.md) implements
that next controlled test with the same initializer, optimizer settings and200
updates per arm, using ten distinct non-probe positions per train sequence.
It is a separate experiment; neither of the heads in this report initializes it.

Actual report: `experiments/registration_alignment_violation_cpu_01/report.json`,
SHA256 `bf64bfdd3eee59c4f167781c6b1d040bfea8843caf48f9cdd159adb59ee11ee9`.
Diagnostic head SHA256 values:

- Geometry only: `99baa6091f215d70a451283cef5488008037d57578549120435847411e40f80b`.
- Geometry plus violation: `410e920c5e581f53d23a8e6ae532ef425b60f61cfcbe7d0e637bcdac92895a11`.

## Reproduction

From the repository root, using a fresh output directory:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/aero-registration-mpl \
  .venv/bin/python -u -m scripts.probe_registration_alignment_violation \
  --config configs/experiment/registration_alignment_violation_cpu.yaml \
  --out-dir experiments/registration_alignment_violation_cpu_repeat01
```

This command uses CPU only. Actual run results must come from its `report.json`,
not from the existence of this command or a passing synthetic test. Even an
improved result cannot approve physical registration, registered conditioning
export or generator training.
