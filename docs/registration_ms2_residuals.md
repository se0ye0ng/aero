# MS² residual attribution and intrinsic sensitivity

Status: 2026-09-24. CPU diagnostics completed. **Not registration qualification,
not generator approval, and not an Anti-UAV300 repair.** This follows the
[fixed-rotation confirmation](registration_ms2_stereo_depth.md).

## What depth alone cannot explain

The residual analysis retains all 60 planned matching cases, all three previously
frozen camera candidates, and the same original-calibration-supported points.
No match is removed for its error, confidence, depth range or image position.
The partitions below describe errors; they do not define an approved subset.

For a fixed pinhole camera pair and source pixel, changing source depth moves
the projected target along an epipolar line. We decompose the native thermal
pixel residual along and perpendicular to that line. A perpendicular residual
over 3 pixels cannot be brought within 3 pixels by changing depth alone while
keeping the cameras and observed match fixed. This does **not** prove that the
match, rather than calibration, motion compensation or another assumption, is wrong.

An additional analytic sensitivity probe permits each RGB stereo disparity to
vary by ±1 pixel and chooses the best target projection along the resulting
segment. This uses the observed match to choose the most favorable disparity:
it is an **optimistic bound**, not a deployable correction or an estimated
physical uncertainty interval. No depth/disparity artifact is changed.

| Matcher, own fixed rotation | Fixed supported points | PCK3 failures | Failures impossible to fix by depth alone | Actual PCK3 | Optimistic ±1 disparity-pixel PCK3 |
|---|---:|---:|---:|---:|---:|
| XoFTR640 | 12,511 | 5,534 | 4,022 (72.68%) | 54.39% | 59.15% |
| MINIMA-XoFTR | 10,703 | 3,461 | 2,271 (65.62%) | 67.22% | 72.29% |

PCK3 is the equal-frame average over all 15 frames, in native thermal pixels.
Counts and the percentage of failures are pooled point counts, not frame rates.
No invalid line geometry or disparity interval occurred in these paired cases;
the implementation retains and counts them if they do occur.

The errors are spatially nonuniform. With the MINIMA-fitted rotation, MINIMA's
upper-middle third has 87.49% PCK3, versus 37.81% in the upper-left and 46.27%
in the upper-right, with all 15 frames represented in every cell. This does
not authorize cropping away the sides. Estimated depth below 10 m has 46.76%
PCK3, compared with 70.79% above 40 m. These associations do not isolate causality:
depth, scene content and image position can be correlated.

High-confidence matches are fewer. MINIMA's confidence ≥0.75 bin contains only
23 supported points in 7/15 frames. It cannot replace the full population as
qualification evidence. Empty strata remain explicit in the report.

Report: `experiments/ms2_residual_attribution_02/report.json`, SHA256
`561e2b7fbbd00d4638093324a757912898df3274f2e340a953a719f9a3fdd837`.
`_01` has identical numerical rows but precedes a lint-only source line wrap;
use `_02` for the current source identity.

## Targeted camera-intrinsic sensitivity experiment

The spatial pattern motivates testing a four-parameter effective thermal
intrinsic correction: independent focal changes limited to ±2%, and principal
point changes limited to ±3 native thermal pixels. These are fixed diagnostic
perturbation bounds, **not measured calibration uncertainty**. No distortion
field, per-frame transform, additional rotation or source depth is fitted.

Each model uses its own previously frozen rotation and the original even-indexed
8 training frames. All originally supported match points enter the soft-L1
objective with equal per-match weights and a 1-pixel scale. The old 8 check frames
and the already observed 15-frame confirmation panel do not enter optimization.
However, the hypothesis was motivated by the latter's residuals, so its scores
below are **exploratory reuse, not a new unseen confirmation**.

Both fits converged without hitting a bound:

| Fit source | Focal-x change | Focal-y change | Principal-x shift | Principal-y shift |
|---|---:|---:|---:|---:|
| XoFTR640 original fit8 | −1.3511% | −0.7713% | −0.0487px | −0.4036px |
| MINIMA original fit8 | −1.5086% | −1.1810% | −0.0490px | −0.5776px |

| Evaluation model, own correction | Original calibration PCK3 | Rotation-only PCK3 | Rotation + intrinsic PCK3 | Rotation / augmented median error |
|---|---:|---:|---:|---:|
| XoFTR, previous check8 | 33.96% | 54.21% | 61.32% | 2.626 / 2.571px |
| MINIMA, previous check8 | 40.82% | 64.02% | 71.23% | 2.225 / 2.235px |
| XoFTR, observed confirmation15 | 21.59% | 54.39% | 61.36% | 2.544 / 2.103px |
| MINIMA, observed confirmation15 | 30.42% | 67.22% | 74.95% | 2.026 / 1.615px |

All scores retain the same original support for each panel/model. Median error
means the median of per-frame medians. The check8 MINIMA median slightly worsens
even though its PCK3 improves; the two statistics must not be conflated.

On the observed 15-frame panel, own augmented corrections improve 15/15 XoFTR
frames and 14/15 MINIMA frames relative to rotation-only. The single MINIMA
regression is retained. No candidate loses projected support. Wrong-image PCK3
is 0.307% for XoFTR and 0% for MINIMA. Cross-evaluation is also saved: the
XoFTR-fitted augmented camera yields 75.23% on MINIMA's points, and the
MINIMA-fitted one yields 60.89% on XoFTR's points. Neither cross-score selects
a qualified camera, and both matchers share an architecture.

Report: `experiments/ms2_intrinsic_sensitivity_01/report.json`, SHA256
`d71ad7849fbc26bc1395b5475ed79978099bd148399a19a039df69c58b40e2ff`.
Preflight SHA256:
`8325473990fd7c49ec74a791e3f0601f93a6efff28ef7f10b0c7312b81081156`.

## Interpretation and next boundary

Depth-only replacement is insufficient under the current camera/match model.
A small global camera perturbation explains some of the spatial residual, with
similar fitted focal changes for two checkpoints. This is evidence for a useful
**effective camera candidate**, not proof that the author's physical calibration
is wrong: the fit can absorb matcher biases, residual motion or rectification
errors. Raw author calibration and all source images remain unchanged.

Before treating the augmented camera as confirmed, freeze its parameters and
test a new panel not used to form this hypothesis. Dense visibility/support and
independent image-based checks remain necessary before any unmasked generator
export. The Anti-UAV300 qualification failure remains separate and unresolved.

The [next confirmation protocol](registration_ms2_intrinsic_confirmation.md)
now freezes all five cameras on15disjoint quarter-interval frames. Extraction
and CPU evaluation have completed: own augmented PCK3 is57.60%(XoFTR) and
68.11%(MINIMA), improving15/15frames relative to own rotation-only. This
confirms an incremental improvement, not full registration qualification.

The new code has 11 targeted tests, including known intrinsic recovery, bound
enforcement, epipolar decomposition, disparity intervals, invalid geometry and
empty-bin denominators. All 130 related MS²/geometry tests passed, followed by
the full CPU suite: **840 passed, 5 existing dependency warnings**. Repository
Ruff and `git diff --check` passed. No GPU training was launched.

Optional CPU reproduction (fresh destinations; do not overwrite completed runs):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.analyze_ms2_residuals \
  --out experiments/ms2_residual_attribution_repeat01/report.json
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_intrinsic_sensitivity \
  --out-dir experiments/ms2_intrinsic_sensitivity_repeat01
```

## Joint rotation and intrinsic optimization

A subsequent CPU experiment tests whether freezing rotation before fitting
intrinsics was a material limitation. It jointly optimizes the same seven
parameters on the original eight fitting frames: rotation-vector components
bounded to ±3 degrees, focal changes to ±2%, and principal-point changes to ±3
native thermal pixels. These remain diagnostic bounds, not measured physical
uncertainty. Soft-L1 loss, equal match weights and original support populations
are unchanged. Two initializations (author zero and the previous sequential fit)
are compared using fitting cost only, never evaluation scores. No frame or match
is rejected, and all wrong-pair controls are retained.

| Model and observed quarter panel | Sequential PCK3 | Joint PCK3 | Sequential median | Joint median |
|---|---:|---:|---:|---:|
| XoFTR640 | 57.60% | 57.81% | 2.411 px | 2.447 px |
| MINIMA-XoFTR | 68.11% | 68.45% | 1.802 px | 1.756 px |

PCK3 is the equal-frame mean on the unchanged author-supported points; medians
are medians of frame medians. These evaluation frames were already observed,
so this is exploratory comparison rather than another unseen confirmation.
Original check8 PCK3 also changes only slightly: 61.32% to 61.71% for XoFTR and
71.23% to 71.72% for MINIMA. Wrong-pair quarter-panel PCK3 remains 0.133% and
0.370%, respectively. All 868 model/frame/condition/candidate rows are retained.

Both joint fits converge but hit the negative vertical principal-point bound
(-3 px). Their normalized robust-Jacobian condition numbers are approximately
213 and 205; these are optimizer diagnostics, not uncertainty intervals.
The two starts converge to essentially identical fitting costs. The small gains
do not support sequential fitting as the principal explanation of the remaining
registration problem, nor do they justify widening limits after seeing results.
No fitted camera is installed or promoted to physical calibration.

Report: `experiments/ms2_joint_camera_01/report.json`, SHA256
`14241d3bafd9624d516494d855c95e67729d3d0b33f68c8bfbeb1092b2d77b86`.
A fresh repeat produced an identical report byte for byte. All 739 input/source
identities were checked, and 300 existing quarter-panel baseline score rows were
reproduced exactly. Five new tests cover known-parameter recovery, parameter bounds
and invalid depths; the full CPU suite passed **962 tests**, with five existing
warnings. Registration and generator authorization remain on HOLD.

## Reverse matching consistency

All 60 quarter-panel model/frame/condition cases were matched again with inputs
reversed: IR-to-RGB rather than RGB-to-IR. The two checkpoints, thermal display
window, image resizing and native-coordinate restoration are unchanged. No camera
is refitted. A forward match is reciprocal only if a reverse match agrees at both
endpoints within the declared tolerance in each endpoint's native pixel coordinates.
The fixed tolerances are 1 and 3 px; neither is a physical-accuracy gate.

| Model and paired cases | All forward matches | Reciprocal at 1 px | Original own-camera PCK3 | Conditional reciprocal PCK3 | Reciprocal and within 3 px on original reference |
|---|---:|---:|---:|---:|---:|
| XoFTR640 | 18,028 | 15,495 | 57.60% | 58.71% | 51.17% |
| MINIMA-XoFTR | 14,592 | 12,548 | 68.11% | 69.09% | 59.80% |

Scores use each model's previously frozen rotation-plus-intrinsic camera, not the
new joint fit. PCK3 columns are equal-frame means over all 15 frames. Conditional
scores use only reciprocal, author-supported points (11,536 XoFTR and 9,565 MINIMA);
the last column keeps the original author-supported denominator (13,271 and
11,076). Thus removing nonreciprocal points cannot silently create a higher full
qualification rate. At 3 px reciprocity tolerance, conditional PCK3 remains only
58.65% and 68.96%.

The wrong-image controls are particularly important: 2,073/2,777 XoFTR matches
and 283/382 MINIMA matches remain reciprocal at 1 px, approximately 74.6% and
74.1%, despite unrelated images. Their conditional median-of-frame camera errors
are 83.81 and 88.34 px. Reciprocity is therefore not an independent correctness
certificate. The implementation also uses dual-softmax and mutual matching inside
the pretrained architecture, so reversing the same model is not a separate
measurement system.

**Decision:** reciprocal selection does not explain away the remaining spatial
disagreement or supply qualified reference points. Do not use this subset as
pseudo-GT, lower the original denominator, or authorize generator training from it.
This rules out another proposed consistency shortcut; it does not prove automatic
registration impossible or identify which camera/matcher assumption is wrong.

Report: `experiments/ms2_reciprocity_01/report.json`, SHA256
`ac0517168117e09eeedfbe2f4ab858eab9ee01f00aee1ebf47b0d4cb964ae641`.
Verification checked 738 input identities and 60 artifact hashes, rebuilt both
endpoint masks using independent all-pairs distances, reproduced 300 original
baseline score sets, and reconstructed all 600 conditional/full-reference score
pairs. The 23 related matching/confirmation tests pass, as do Ruff code checks.
The latest full CPU suite remains **962 passed**. No GPU inference or training was
launched, and no scientific approval changed.
