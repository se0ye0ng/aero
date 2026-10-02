# Separating global geometry from image-match quality

Executed on2026-09-24. These are CPU diagnostics, not registration qualification.
Original training products and engineering thresholds are unchanged.

## Reference-assisted capacity check

`configs/experiment/registration_external_geometry_cpu.yaml` freezes the prior
four-pair reference manifest and image-only report by hash. Five-fold
cross-fitting predicts every reference point exactly once per partition seed
(0/1/2). Fitting references and scored references are disjoint within each fold.
This is within-scene point cross-validation, not new-scene generalization.

Affine least squares, non-RANSAC homography and TPS fit RGB→thermal reference
coordinates directly. TPS uses normalized coordinates, degree1 and zero
smoothing through SciPy `RBFInterpolator`.
[Official implementation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html).
These transforms use annotations and are **oracle diagnostics**, never image-only
method results. Direct fitting also differs from inverting the earlier
image-fitted thermal→RGB homography. Do not infer an exact error decomposition.

| Reference-assisted family | Equal-pair mean PCK at3 thermal file pixels, range across partition seeds |
|---|---:|
| Affine | 26.91–28.83% |
| Homography | 26.28–30.22% |
| TPS | 83.19–85.26% |

No fold failed numerically. The difference supports inadequate global geometry
on these four scenes, rather than interpreting every homography error as a bad
match. It does not separate parallax from annotation uncertainty, prove a noise
floor, or establish an invertible dense warp. TPS extrapolation is permitted in
this reference-capacity check, unlike the image-only candidate below.

Report: `experiments/registration_external_geometry_cpu_01/report.json`, SHA256
`4e143bcc2fa6c3d3f1cb69d01628613552786b6262041b9b3b94843ed8dfb28a`.

## Image-only local-consensus TPS candidate

`configs/experiment/registration_external_local_warp_cpu.yaml` defines one
exploratory pipeline, frozen before its scores were computed. It reuses hashed
XoFTR/LoFTR matches; reference points are read only after the warp is fitted.

1. Keep the highest-confidence match at a duplicated RGB coordinate.
2. For each match, fit local affine relations to its32 nearest RGB neighbors,
   excluding the match itself. Four trimming iterations retain75% of neighbors.
3. Keep matches whose own predicted thermal coordinate agrees within
   `max(2px, 0.005 × thermal_file_long_side)`.
4. Keep the highest-confidence accepted match per12×12 RGB image-grid cell;
   require at least six controls. Fit TPS with smoothing0.0001 in normalized
   coordinates.
5. Reject queries outside the controls' convex hull or outputs outside the
   thermal frame. Unsupported references remain failures in every PCK denominator.

The pipeline changes filtering, control distribution and warp family jointly;
this is not an isolated architecture ablation. It has no topology or inverse-
consistency guarantee and must not be directly exported as qualified conditioning.

| Image-only method | Global homography: equal-pair PCK3 | Local TPS: equal-pair PCK3 | Unsupported references with local TPS |
|---|---:|---:|---:|
| XoFTR | 36.10% | 84.07% | 26/241 |
| LoFTR | 37.33% | 69.94% | 39/241 |

Every pair and reference stays in the denominator. PCK uses stored thermal file
pixels, not a common angular unit; one thermal file is1280×1024, the other three
640×512. The panel was already examined during preceding diagnostics, so this is
exploratory evidence, not confirmation on unseen data or a paper-ready seed study.
Global RANSAC seeds0/1/2 were identical; this deterministic postprocessor does not
create independent training repetitions.

Report: `experiments/registration_external_local_warp_cpu_01/report.json`, SHA256
`0c1d8e35cebdc67f507cf94edc3f2dd7a77782bd2cb149dda36f6394f15d86d8`.

## Actual Anti-UAV transfer check

The unchanged candidate settings were then applied to the existing hashed16
train pairs in `xoftr_overlay_train16_01/report.json`. This avoids another neural
inference/training run. Matching coordinates remain on the recorded640-wide
working grids; cropped-input matches already have their header offsets restored.
Annotations are rehashed and converted to the same grid **after** fitting.
Reverse fitting uses the same rule in target RGB working pixels.

| Anti-UAV condition | RGB→IR warp fits | Both warped box-corner envelopes available | Both-direction corner-IoU≥0.6 |
|---|---:|---:|---:|
| Header matches removed after inference | 5/16 | 3/16 | 1/16 |
| Header removed before inference | 6/16 | 4/16 | 1/16 |

These are box-corner proxies, not physical point errors or the full v4 joint
gate. Nonlinear box interiors may extend beyond their corner envelope. Fits in
the two directions are separate and not guaranteed inverses. No failure is
discarded, including the one pair whose target intersects the removed header.

In the input-header-cropped forward case, two pairs have insufficient unique
matches and eight have insufficient selected controls. Importantly, the latter
is **not proof that corresponding information is absent**: one case has32
accepted matches reduced to four grid controls. The global grid/coverage rule
can discard densely clustered target information. Other cases have zero local-
consensus matches. These are distinct causes and should not be conflated.

Report: `experiments/registration_antiuav_local_warp_cpu_01/report.json`, SHA256
`5962d0ab64678ef47111a93fc02f9591a98be88cbb0c750d1e23f56c71957624`.

## Decision and reproduction

Local image-conditioned geometry is a promising external-data result, but this
unchanged candidate **does not transfer sufficiently to Anti-UAV**. Do not launch
a full training run, replace the original model, or approve a generator from it.
The control-selection follow-up below tests whether preserving small clustered
targets is sufficient. It is not: more available fits did not increase joint
box-proxy passes. Unsupported/failing observations remain in every denominator.

All commands use CPU and fresh output directories:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.analyze_external_landmark_geometry \
  --out-dir experiments/registration_external_geometry_cpu_repeat01
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_external_local_warp \
  --out-dir experiments/registration_external_local_warp_cpu_repeat01
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_antiuav_local_warp \
  --out-dir experiments/registration_antiuav_local_warp_cpu_repeat01
```

Only project-owned code, configurations and aggregate diagnostics are candidates
for Git release. External raw data and reference points remain ignored. The
original registration, controlled downstream experiments and public-release
requirements are still incomplete.

Verification after all three diagnostics:691 CPU tests passed, repository Ruff
and `git diff --check` passed, and every recorded input/source hash in the three
reports was rechecked. These checks establish implementation/artifact integrity,
not scientific qualification or GitHub publication.

## Control-selection follow-up: completed, not adopted

`configs/experiment/registration_control_selection_cpu.yaml` freezes the two
preceding reports by hash. The new reusable engine first reproduces the saved
grid controls, fit status, external per-point errors and Anti-UAV directional
IoUs (absolute numeric tolerance1e-8). It then changes only the selection policy:
keep all accepted points when there are at most144; otherwise start from the
highest-confidence point and select farthest points in RGB/source coordinates.
Consensus filtering, TPS smoothing, support rejection and scoring are unchanged.
The maximum144-control budget is matched, **not the actual number of controls**.
No annotation is used in selection or fitting.

| External image-only method | Grid PCK3 | Farthest-point PCK3 | Grid PCK5 | Farthest-point PCK5 |
|---|---:|---:|---:|---:|
| XoFTR | 84.07% | 80.17% | 87.00% | 89.50% |
| LoFTR | 69.94% | 68.26% | 78.18% | 84.03% |

These are equal-pair means on the same previously examined four pairs. The mixed
threshold effects do not justify choosing a more favorable reporting threshold.

| Anti-UAV condition | Grid RGB→IR fits | Farthest-point RGB→IR fits | Grid joint box passes | Farthest-point joint box passes |
|---|---:|---:|---:|---:|
| Header postfilter | 5/16 | 9/16 | 1/16 | 1/16 |
| Input header crop | 6/16 | 8/16 | 1/16 | 1/16 |

Report: `experiments/registration_control_selection_cpu_01/report.json`, SHA256
`645b9b217968af95e1a8bf0cfed22720b5d1ccc3bcf41111b3699403ba34333b`.
The exact grid replay passed. Farthest-point selection is not promoted as a
repair; counting a fitted function as successful registration would be incorrect.

## Target-support attribution

A separate read-only diagnostic inspects those saved fits and matching arrays.
Only after fitting, annotation boxes identify where matches and controls occur;
they never refit or select a transform. Source-box corners are checked against
the convex hull of the exact saved controls. Unsupported corners remain failures.

For input-header-cropped matching with farthest-point controls:

| Direction | No fitted warp | Source corners outside control hull | Available but IoU below0.6 | Directional box pass |
|---|---:|---:|---:|---:|
| RGB→IR | 8/16 | 4/16 | 3/16 | 1/16 |
| IR→RGB | 7/16 | 4/16 | 2/16 | 3/16 |

All16 pairs are represented per direction; only one passes both. The two newly
available forward fits remain unsupported at box corners. Even among supported
fits, incorrect target extent/location remains. Thus neither a relaxed hull rule
nor merely retaining more controls is an established correction.

Before any filtering, four of16 forward cases have no match in the source target
box, and five have no match lying in both corresponding boxes. Reverse counts
are three and six. Both-box membership is **not pixel correspondence truth**;
absence in this frozen inference is not proof the images cannot be matched.
This localizes several distinct failure stages rather than assuming more epochs
will repair them. Any next matching diagnostic must preserve the original panel,
avoid GT-guided cropping/fitting and test supported accuracy as well as coverage.

Report: `experiments/registration_target_support_cpu_01/report.json`, SHA256
`936f6c0f0502b8ec15ea57bf5ec6f42440972c57d59c048e2092d48edd3a48dc`.

Reproduce on CPU using fresh output directories:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_control_selection \
  --out-dir experiments/registration_control_selection_cpu_repeat01
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.analyze_registration_target_support \
  --report experiments/registration_control_selection_cpu_repeat01/report.json \
  --out-dir experiments/registration_target_support_cpu_repeat01
```

The complete CPU suite passed with702 tests after both changes; repository Ruff
and `git diff --check` passed. All103 and105 recorded input/source hashes in the
two new reports were rechecked, respectively. Qualification, generator approval
and release status are unchanged. No additional GPU training was launched.

The subsequent [native-derived RGB-resolution comparison](registration_rgb_resolution.md)
also completed: doubling RGB input width retained the same 1/16 joint box-proxy
pass for each selection policy. This is a tested negative alternative, not a
pending GPU command or a claim that higher detail can never help another method.

## Sampled geometry of the external local warps

The frozen UAV-TIRVis local TPS fits were checked on a native RGB grid at
16-pixel spacing. Central differences at ±1 RGB pixel estimate the Jacobian;
a separate reverse TPS, fitted with the same policy on swapped matches, measures
round-trip error. References do not guide either fit. Unsupported grid locations
remain in the coverage denominator.

| Method | Scene | Forward grid support | Round-trip median | Round-trip p95 |
|---|---|---:|---:|---:|
| XoFTR | Mountain | 33,881 / 47,628 | 2.989 px | 13.066 px |
| XoFTR | MountainResort | 21,192 / 46,750 | 2.570 px | 9.660 px |
| XoFTR | ResidentialArea | 22,571 / 46,750 | 1.640 px | 4.437 px |
| XoFTR | Seaside | 24,074 / 46,750 | 1.544 px | 8.996 px |
| LoFTR | Mountain | 30,563 / 47,628 | 6.286 px | 17.446 px |
| LoFTR | MountainResort | 21,958 / 46,750 | 3.446 px | 14.847 px |
| LoFTR | ResidentialArea | 22,819 / 46,750 | 2.160 px | 7.092 px |
| LoFTR | Seaside | 22,120 / 46,750 | 3.528 px | 11.360 px |

Round-trip statistics use only supported round trips and are in **native RGB
pixels**, not the thermal pixels used by landmark PCK. The reverse TPS is not
a guaranteed mathematical inverse. No nonpositive Jacobian determinant was
observed at the 101,559 supported XoFTR and 97,336 supported LoFTR derivative
locations. This sampled result is not an exhaustive no-folding or global
injectivity certificate. XoFTR's source-grid support is only 45.33–71.14% across
the four scenes. Thus the earlier landmark improvement does not establish
full-frame, unmasked generator supervision or correct correspondence everywhere.

Report: `experiments/registration_external_warp_geometry_01/report.json`, SHA256
`80fcfc4784a2dc50c7a95fb1e39abc02b0af0a27d3cb2a4f8079dca8eea9c911`.
The audit records 33 input/source identities and eight array artifacts.
Registration qualification and generator authorization remain false.

## External resolution comparison

The completed GPU diagnostic compares XoFTR input long-side caps of 640 and 1280 on
the same four external pairs, with unchanged TPS policy and all 241 reference
landmarks retained per cap. Images are not upsampled: three native 640-pixel
thermal images remain at that resolution. This is inference, not training or
an Anti-UAV transfer result. It differs from the completed negative Anti-UAV
RGB-resolution test above. Inference ran on an RTX 3090. CPU verification checked
all eight saved cases, source identities, artifact hashes, recomputed TPS fits,
individual reference errors, summaries and aggregates. This is saved-result
verification, not a second GPU inference replay.

| Scene | PCK3 at cap 640 | PCK3 at cap 1280 | Unsupported at 640 | Unsupported at 1280 |
|---|---:|---:|---:|---:|
| Mountain | 81.82% | 81.82% | 5/44 | 1/44 |
| MountainResort | 79.41% | 55.88% | 6/34 | 6/34 |
| ResidentialArea | 89.13% | 53.26% | 10/92 | 34/92 |
| Seaside | 84.51% | 16.90% | 5/71 | 3/71 |
| Equal-pair mean | **83.72%** | **51.97%** | — | — |

PCK3 uses native thermal pixels and includes every reference landmark in its
denominator. Seaside's supported-point median error increases from 1.057 to
51.997 pixels despite fewer unsupported points. The deterioration therefore
cannot be explained solely by loss of coverage. Neither the higher match count
in some scenes nor greater input resolution establishes better correspondence.
The 1280-cap setting is not promoted. This experiment does not qualify the
640-cap setting either, and it cannot replace Anti-UAV validation.

Report: `experiments/registration_external_resolution_gpu_01/report.json`, SHA256
`2e87d1b7ad9c86eea43c75de2e65aa1885e7c3d78674eeafca05b9d73c055a92`.
The original CPU-only 640 result and this GPU 640 result are distinct inference
runs; their scores must not be presented as bit-identical replay results.

To verify the completed artifacts without GPU inference:

```bash
cd <repository root>
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_external_resolution verify \
  --out-dir experiments/registration_external_resolution_gpu_01
```

The GPU launcher refuses to overwrite this completed run. Preserve it; do not
rerun the default command as a way of verifying saved results. Qualification and
generator authorization remain false.

## Boundary control confirmation

The boundary candidate preserves every original grid control and adds convex-hull
vertices from matches passing the unchanged leave-one-out local affine consensus.
Neither reference coordinates nor reference errors choose controls. TPS smoothing,
input cap 640, confidence thresholds and unsupported-point handling are unchanged.
This addresses loss of source support during grid reduction, not the correctness
of every accepted correspondence.

The candidate and additional pair IDs were frozen before confirmation inference.
Confirmation uses IDs 2, 3 and 4 in each of the four previously observed scenes.
It is not unseen-scene generalization. Seaside/2 has the same two landmark files
as development Seaside/1; it remains in the primary result and is also removed in
an explicitly labelled sensitivity analysis. The 11-pair result is not claimed
to contain statistically independent scenes or frames.

| Panel | Pairs | References | Grid PCK3 | Boundary PCK3 | Grid unavailable | Boundary unavailable |
|---|---:|---:|---:|---:|---:|---:|
| Development | 4 | 241 | 83.72% | 87.03% | 26 | 12 |
| Additional same-scene pairs | 12 | 499 | 63.37% | 67.89% | 107 | 66 |
| Additional pairs without duplicated development labels | 11 | 428 | 61.71% | 66.25% | 99 | 62 |

PCK3 is the equal-pair mean at three native thermal pixels; every reference,
including unavailable predictions, stays in each pair's denominator. Eight of
12 confirmation pairs improve, three tie and one worsens. Confirmation therefore
supports a modest coverage-related benefit, not registration qualification.
The much lower absolute scores on additional pairs prevent presenting the
development score as a general accuracy result. In development, median errors on
common supported points do not improve, so this is not uniformly better alignment.

Two intervening alternatives did not resolve the problem. Replacing TPS with
linear interpolation on the same controls gives PCK3 82.73% at cap640 and 54.35%
at cap1280. Greedy removal of controls implicated in nonpositive triangle
orientations gives 81.49% and 52.24%, while unavailable landmarks increase to
36/241 and 105/241. Positive control-triangle orientation is not a dense TPS
Jacobian certificate and does not imply correct correspondence.

Transferring the fixed boundary rule to Anti-UAV train16 gives **1/16 joint
box-corner passes both before and after augmentation**, under both header
postfilter and input-header-crop conditions. These are annotation proxies, not
independent pixel accuracy. External coverage improvement does not override this
negative transfer result or authorize generator supervision.

Evidence, with no raw dataset redistribution:

- Development: `experiments/registration_external_boundary_controls_01/report.json`,
  SHA256 `8f7a82429888cbd06f193811b89b1c876c97baf8e2b3f28e57c6441d261ce42a`.
- Confirmation: `experiments/registration_external_boundary_confirmation_01/report.json`,
  SHA256 `0d6ad329469cc8126715e2dd707bd87455de8e835a05b6645488cbf1ec78491e`.
  All 158 recorded input/source identities and 12 artifacts were checked, and both
  transforms, reference errors and aggregates were reconstructed from saved matches.
  This does not replay neural inference or quantify landmark annotation uncertainty.
- Anti-UAV transfer: `experiments/registration_antiuav_boundary_controls_01/report.json`,
  SHA256 `ed7d7584d1ce83d7391927d467d1e48ca012994d955a9d26d7c3a67647384648`.

Reproduce confirmation on CPU with a fresh destination and the pinned input cache:

```bash
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -u -m scripts.probe_external_boundary_confirmation \
  --out-dir experiments/registration_external_boundary_confirmation_repeat01
```
