# Fast inverse-consistency investigation

The existing v4, v5 and v6 checkpoints were frozen. No optimizer updates were
performed. The probe uses official train-cache frames only, selecting evenly
spaced sorted sequence IDs and the midpoint usable cache frame in each sequence.
It records the selected decoded array hashes rather than rehashing whole video
files or cache shards. Cache-position IDs must not be interpreted as video IDs.

Five variants are measured under the unchanged v4 geometry definitions:

1. Original independently predicted reciprocal fields.
2. Keep the native IR-to-RGB point map; numerically invert it for RGB-to-IR.
3. Keep the native RGB-to-IR point map; numerically invert it for IR-to-RGB.
4. Fit a global affine map to uniformly sampled predictions from the first field,
   then use its analytic inverse.
5. Repeat the affine projection using the opposite field.

Numerical inversion uses 20 damped Newton iterations, a 0.05 normalized-coordinate
step cap and line search (1, 0.5, 0.25). Its initializer is the learned reciprocal.
No annotations enter either inversion or affine fitting. An inverse need not exist
for folded or out-of-view points; original support/topology/ROI checks still apply.
Cycle gains from constructed inverses are expected algebraically and cannot
establish physical correspondence accuracy.

## Initial 16-sequence train probe

Both checkpoint probes took about 21 seconds of CPU computation after loading.
For v6, native IR-to-RGB plus numerical reciprocal raised joint engineering pass
from 0/16 to 10/16. Its forward median box IoU remained 0.797 (identical forward
field) and reverse median IoU was 0.727 versus 0.760 for the learned reciprocal.
Cycle p95 medians became 0.00815 / 0.00000381 network pixels. The opposite-anchor
inverse achieved 6/16. V4's first-anchor inverse achieved 4/16. Neither affine
projection passed a frame; median box IoU was near zero. This rules out this
particular projection, not every affine registration architecture.

The original v6 cycle error also appears inside the image: median interior p95
3.88 / 3.81 px, versus 3.80 / 4.21 px in the outer 16-pixel band. A border-only
explanation does not account for these samples. The small probe selected the
native IR-to-RGB anchor for the next fixed comparison; it is not qualification.

## Expanded probe and stopping decision

The v6 probe was expanded to all 160 official train sequences, one midpoint usable
pair per sequence (160 pairs, not all 141,816 training pairs). CPU compute time was
175.4 seconds. Results under the unchanged engineering criteria:

| Variant | Joint pass | Median IoU IR-to-RGB / RGB-to-IR | Median global cycle p95 px |
|---|---:|---:|---:|
| Original v6 | 0/160 | 0.7723 / 0.7722 | 3.689 / 3.779 |
| Keep IR-to-RGB, invert | 119/160 (74.375%) | 0.7723 / 0.7366 | 0.00804 / 0.00000381 |
| Keep RGB-to-IR, invert | 101/160 (63.125%) | 0.7056 / 0.7722 | 0.00000381 / 0.00704 |
| Either global affine projection | 0/160 | about 0.01 / 0.01 | about 0.00001 / 0.00001 |

Artifacts: `experiments/registration_inverse_probe_v6_train160_01/report.json`,
plus the two `registration_inverse_probe_{v4,v6}_train16_01` directories.
These constructed cycle improvements do not provide physical correspondence GT.

The existing 536-frame validation comparison also gives a conditional upper bound:
if the native IR-to-RGB field remains fixed, **no reciprocal-only change can exceed
456/536 = 85.075% joint pass**. The opposite anchor is bounded by 458/536 = 85.448%.
The bound counts only mandatory conditions that depend on the fixed field: box
IoU, centroid, area, bounds, native support and global/ROI positive Jacobian.
It excludes all conditions that might improve when the reciprocal is replaced.
In the first direction, 63 frames already fail IoU and 36 fail area (overlapping
failure groups). These cannot be repaired by changing the reverse field alone.
At least 54 additional frames would need their fixed-field constraints repaired
to reach 510/536 (the 95% threshold), even assuming a perfect reciprocal.

This is a bound for these checkpoints and this intervention, not an impossibility
claim about registration in general. Script:
`scripts/registration_inverse_upper_bound.py`; saved result:
`experiments/registration_inverse_upper_bound_01.json`.

**Decision:** stop inverse-only qualification attempts and whole-field affine
projection exploration. Do not spend a long training run or a GPU screen to try to
clear a threshold that this fixed-map intervention cannot reach. The next viable
model intervention must change native alignment as well as enforce reciprocal
consistency; simply increasing cycle loss or reconstructing the inverse is
insufficient. A shared invertible map trained against both-direction geometry is
an unvalidated next candidate, not implemented or claimed solved here. Independent
correspondence evidence still remains a separate prerequisite for generator use.

## Optional GPU screen (diagnostic only; not needed for the stopping decision)

```bash
bash scripts/run_antiuav300_inverse_screen.sh
```

This evaluates v4, original v6 and the selected v6 numerical-inverse adapter on
the identical 1,280 train and 536 validation frames. Original checkpoints and
qualification thresholds stay fixed. Output:
`experiments/antiuav300_v6_inverse_screen_01/{summary.md,summary.csv,report.json}`.

This command was prepared before the validation upper bound was established. It
is retained for reproducibility, not recommended as a way to obtain a PASS.
The wrapper preserves the scheduler GPU mask. Use `--preflight-only` to verify
inputs without inference. For a repeat, choose a new directory with
`AERO_INVERSE_SCREEN_OUTPUT`. No GPU screening was performed during preparation.

If the candidate fails, use per-frame failures to separate residual box accuracy,
support, inversion convergence and ROI topology. Do not infer that increasing
epochs will fix these. If it passes the sampled geometry screen, exhaustive
evaluation and independently reviewed correspondence/occlusion evidence remain
necessary before generator training. Exit zero means diagnostic completion only.
