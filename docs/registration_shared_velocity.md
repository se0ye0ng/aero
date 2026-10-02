# Shared-velocity initialization: implemented and checked on real train frames

## Why this experiment

V6 still predicts two independent displacement fields. Replacing only the inverse
cannot meet the frozen geometric gate because fixed-forward box failures impose
an85.075% upper bound on the existing validation screen. A candidate must change
native alignment and its reciprocal jointly, without treating low cycle error as
physical accuracy.

`src/aero_ir/registration/shared_velocity.py` now implements scaling-and-squaring
of one stationary velocity: the maps are approximately exp(v) and exp(-v). This
is established registration machinery, **not a novelty claim**. See the
[official VoxelMorph model implementation](https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/nn/models.py)
and [VoxelMorph registration paper](https://arxiv.org/abs/1809.05231).
The code is an original compact implementation using this project's normalized
pixel-centre convention; no new third-party package is installed or vendored.

Crucially, a displacement is **not** a stationary velocity or the logarithm of a
deformation. The current diagnostic tests the naive reuse of v6's native forward
displacement as v, with seven integration steps. It does not train a new matcher,
preserve the old forward map, fit to boxes, or claim exact discrete invertibility.

## Completed checks

Ten CPU tests cover exact zero-field identity, known translation on valid support,
nonlinear forward/inverse composition, positive Jacobian on a smooth example,
finite nonzero backward gradients, bounded pixel-unit controls, optional boundary
taper, and rejection of invalid/nonfinite inputs. These examples do not guarantee
arbitrary predicted fields are fold-free; actual fields still undergo the unchanged
v4 support/Jacobian/cycle/box checks.

Real-data report: `experiments/registration_shared_velocity_init_train16_01/report.json`.
SHA256: `836441c3b4d6c1ee472c905bd488c451f2e8fd905265413d35966e7dcdb2e179`.
CPU inference/evaluation took19.33 seconds after initialization. Six executed sources
are archived and hash-verified. Checkpoint and cache hashes match the previous
v6 inverse diagnostic; every selected decoded frame/box array hash is checked.
The original v6 per-frame measurements reproduce the old report exactly.
This uses16 fixed train midpoints, not all training frames or held-out evaluation.

| Variant | Joint geometric pass /16 | Median box IoU IR→RGB / RGB→IR | Median global cycle p95, pixels IR→RGB / RGB→IR |
|---|---:|---:|---:|
| Original v6, independently predicted maps | 0 | 0.79725 /0.75983 | 3.94505 /3.96391 |
| Naive shared-velocity initialization, no optimization | 8 | 0.70353 /0.69088 | 0.04500 /0.06027 |

Thus coupling the maps improves numerical cycles but **harms box alignment**.
The50% geometric pass is below the frozen95% requirement. It is also lower than
the previously tested native-forward plus numerical-inverse result of10/16 on
this same panel. Neither comparison establishes physical dense accuracy.
No validation/test data, previous checkpoint, pairing, or gate was changed.

## Reproduce, CPU only

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_shared_velocity \
  --output-dir experiments/registration_shared_velocity_init_repeat01
```

The default parent report freezes the16 sample identities and the checkpoint path.
The full train cache is required, not the16-frames-per-sequence cache; a mismatched
manifest fails before creating an experiment directory. An existing output is never
overwritten. The parent report/cache must remain available for reproducibility.

## Decision and remaining requirements

Do not deploy the naive reparameterization or request another300-epoch run with it.
The next candidate needs a learned or fitted velocity initialization and joint
native-alignment optimization, not a rename of existing displacement tensors.
An image-based correspondence signal must be assessed independently of its training
objective; the SAM-only ablation does not supply that evidence. The new transform
primitive is now used by the [v7 matched-budget training pilot](registration_v7.md).
Its image-conditioned trainer, real-data CPU update smoke and commands are ready;
GPU pilot results and qualified correspondence evidence remain incomplete.

The full objective still includes qualified registration, actual paired-generator
training, matched detector baselines/idea arms, reproducible results, and a reviewed
GitHub release. Neither this structural diagnostic nor the mask non-deterioration
fix completes those requirements. Generator training remains HOLD.
