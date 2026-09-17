# Registration v4: coordinate correction and qualification policy

Status: implementation and CPU regression tests; real-checkpoint GPU audit pending.
This is a new protocol, not a reclassification of v1/v2/v3. Those modules, checkpoints,
reports and running processes are preserved. Do not interpret a historical runner's
`qualified` message as permission to start paired generator training.

## One coordinate contract

All new public fields are `N,2,H,W` normalized backward displacements on the
`align_corners=False` pixel-centre lattice. For pixel index `i` and width `W`:

```
centre(i) = 2 * (i + 0.5) / W - 1
map(p) = p + displacement(p)
```

The legacy SuperFusion architecture and its internal sampling remain unchanged for
checkpoint compatibility. Convert each final raw network output **exactly once**:

```
corrected = raw + endpoint_grid - pixel_centre_grid
```

Thus `centre_grid + corrected == endpoint_grid + raw`: the image warp is preserved,
but boxes, points, Jacobians and reciprocal composition now measure that actual warp.
Raw zero was not identity; corrected zero is identity. Do not change `align_corners`
alone, feed a corrected field to a legacy warp, or apply this adapter twice.

`geometry.py` owns all new geometry. Derivatives use centre spacing `2/W, 2/H`.
Point and cycle validity excludes padding and requires support in both images and on
return. ROI Jacobian stencils must also have valid input support. Nonfinite or absent
measurements fail closed. Finite box grids are diagnostics, not segmentation masks.

Image-warp direction and point-map direction are deliberately distinguished:

| Matcher call | Backward point map | Target-local ROI |
|---|---|---|
| `visible_to_infrared` | IR output -> RGB input | IR box |
| `infrared_to_visible` | RGB output -> IR input | RGB box |

## Corrected training

`protocol_v4.py` receives raw fields, converts them once, and uses the corrected geometry
for both losses. The old point-by-point box perimeter loss is removed: box corners and
perimeter samples are not matched physical landmarks across modalities. Box enclosure,
centroid and area losses remain explicitly weak supervision. Edge NCC uses observed overlap
and excludes Sobel borders; it is a structural proxy, not a correspondence label. Target-local
inverse consistency supplements global consistency; unobserved cycles cannot lower the loss
by being discarded. Smoothness, fold and boundary penalties refer to the corrected field.

The separate v4 trainer retains 300 epochs, 16 rotating pairs/sequence/epoch, batch 8,
96,000 optimizer steps on the official 160 training sequences, and learning rate 5e-5.
It starts from the immutable v2 weights, not a v3 optimizer state. The verified train cache
is reused; validation/test are not opened during training. Source hashes, coordinate
convention and losses are stored in checkpoint metadata. Resume requires matching metadata.
Seeds are controlled, but CUDA grid-sample backward may be nondeterministic: do not promise
bitwise training replay. Epoch counts in metadata are plans; the audit separately requires
the actual checkpoint `epoch == 300`.

## New engineering checks (not directly comparable to old scores)

Thresholds are defined once in `qualification_v4.py`, before the corrected real-data audit.
They are engineering criteria, not validated physical tolerances or literature-derived guarantees.
Each evaluated frame must satisfy **all** checks in **both** directions:

- Box IoU >= 0.6, centre shift <= 0.25 box diagonal, area-ratio change <= 0.5, box in bounds.
- Observed global warp/cycle support >= 90%; global positive-Jacobian fraction >= 99%.
- Global valid-pixel cycle p95 <= 1 pixel (at the evaluated network resolution).
- A filled 9x9 grid over each target box has 100% valid cycle support and positive ROI
  Jacobian samples. This includes tiny boxes with no full pixel centres inside them.
- ROI cycle p95 <= min(1 pixel, 0.10 box diagonal); maximum <= min(2 pixels, 0.25 diagonal).

Both the per-frame joint pass rate and sequence-macro joint pass rate must reach 95%.
Disjoint passing subsets cannot be combined into a pass. The audit verifies the exact expected
set of `(sequence_id, frame_index)` pairs, including no duplicates, and hashes selection inputs.
Each direction's measurements are saved so failures remain inspectable. Screen uses eight
endpoint-inclusive usable pairs per sequence; full uses every usable train/validation pair.
Official test remains unopened. Video-byte hashing is not performed by this audit and is
listed as a report limitation; use the existing dataset provenance for archive integrity.

## Independent physical correspondence: still HOLD

Engineering geometry is necessary, not sufficient. Two incorrect maps can be mutual inverses,
and box agreement does not establish alignment inside a UAV or in the surrounding scene.
The implementation therefore never turns these proxies into generator eligibility:

```
geometric_screen / exhaustive_geometry = measured pass or hold
independent_correspondence             = hold_not_evaluated
dense_paired_image_registration        = hold
generator_training_eligible           = hold
```

Before this HOLD can be lifted, establish a separately versioned, independently reviewed
correspondence panel. Freeze frame selection, reviewer instructions, physical landmark/mask
definitions, visibility/occlusion labels, coordinate units, target/background coverage and
acceptance thresholds **before** evaluating candidate checkpoints. Report correspondence
error in native-image pixels and relative to target size, annotation uncertainty and per-sequence
variation. Exclude unobservable correspondences explicitly rather than using box corners as
ground truth. A target-only panel cannot qualify full-image supervision. Prior validation has
already informed development; final scientific confirmation requires an untouched test or
external paired set after choices are locked. No real landmark annotations or reviewed
correspondence result have been fabricated or supplied by this patch.

## Commands

### Short GPU optimizer smoke (before full training)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_antiuav300_registration_v4_smoke.sh
```

This executes CPU preflight tests, then exactly **3 AdamW updates at batch 8, FP32**
from the same pinned v2 checkpoint and v4 loss/step as full training. The 24 real pairs
come from the official **train split only**, one usable pair per sorted sequence;
this bounded selection is not a representative accuracy benchmark. No validation/test
images are accessed, no complete cache is built, and no weights or optimizer states are saved.
The 300-epoch training schedule is unchanged. `training_v4.training_step` is shared by
smoke and full training to avoid testing a separate optimization implementation.

It checks finite losses, both field gradients, gradient clipping, AdamW state, and actual
nonzero parameter updates; synchronizes CUDA timings and records peak allocated/reserved memory.
Snapshots for update checking increase measured memory and timing, so do not extrapolate
full-training runtime from these three steps. Input checkpoint, source and annotation hashes,
decoded-input hashes, frame selection and Git dirty status are recorded. A dirty checkout is
allowed for this diagnostic, not silently treated as a frozen full experiment.

Result: `experiments/antiuav300_registration_v4_gpu_smoke_v1/smoke_report.json`.
`ok: true` / exit 0 means the engineering smoke passed; **generator eligibility remains HOLD**.
Failure (including CUDA unavailable or OOM) writes `ok: false` / exits 1 once the output
directory is reserved. Invalid arguments/existing output exit 2. Nothing is overwritten;
for a retry set `AERO_REGISTRATION_V4_SMOKE_OUTPUT` to a fresh directory. A killed process
may leave an incomplete directory without a final report; it is not a pass.

Do not run alongside an active training process on the same GPU. GPU smoke success is not
proof of convergence, physical registration accuracy or full 300-epoch memory stability.

### Coordinate audit and full training

Run CPU regressions (no dataset or GPU):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_registration_coordinates_v4.py
```

First re-audit the existing **completed v2** checkpoint; this does not retrain or modify it:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_antiuav300_registration_v4.sh audit
```

The default report is `experiments/antiuav300_coordinates_v4_screen.json`. Existing reports
are never overwritten. To audit a completed v3 checkpoint, set `AERO_REGISTRATION_V4_CHECKPOINT`
to that file and `AERO_REGISTRATION_V4_REPORT` to a new output file. An in-progress `latest.pth`
is rejected. Do not run another GPU job alongside the ongoing v3 run without resource planning.

**Exit 3 after a saved report means research HOLD, not a CUDA or Python failure.** Read
the `gates` and measurements; do not bypass it. Missing inputs or malformed paths fail early.

After reviewing the corrected audit and committing the reviewed protocol, corrected full training
is available separately (not launched by the default command):

```bash
CUDA_VISIBLE_DEVICES=0 \
  AERO_REGISTRATION_V4_REPORT="$PWD/experiments/antiuav300_trained_v4_screen.json" \
  bash scripts/run_antiuav300_registration_v4.sh train
```

It uses a new v4 output directory and never resumes v3. `full` is an explicit exhaustive audit
mode; set a completed checkpoint and a fresh report path. Neither `train` nor `full` clears
the independent-evidence HOLD by itself.

Coordinate reference: [PyTorch grid_sample](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.grid_sample.html).
Inverse consistency motivation: [ICON](https://arxiv.org/abs/2105.04459); this does not validate
our thresholds or the Anti-UAV adaptation.
