# AERO: preliminary measured results

Status: **registration and generated-data benefit are not yet qualified**.
The [machine-readable result snapshot](results_snapshot.json) contains aggregate
measurements and source identities. No native dataset imagery or model weights
are redistributed with this summary. Detection metrics below are percentages;
the JSON preserves their original0–1 fractions.

## Abstract

AERO investigates whether sensor-aware radiometric diagnostics help identify
generated infrared images that are useful for object detection. The intended
comparison holds detector training conditions fixed across real-only,
real-plus-simulated and real-plus-generated data. The completed real-only FLIR
YOLOX-s reference reaches35.56% mAP@0.5:0.95 and57.95% mAP@0.5 on1,144 validation
images. A separate Anti-UAV300 study investigates the registration prerequisite
for paired visible-to-infrared generation. Under a matched3,200-update training
budget, an image-conditioned residual-velocity model passes the frozen geometric
criteria on117 of160 train midpoints, compared with115 for direct predictor
continuation. In the subsequent matched 96,000-update experiment, uniform and
failure-aware sampling pass 125/160 and 124/160 midpoints, respectively; both pass
133/160 quarter-frame pairs. Neither reaches the 95% engineering threshold, and
these screens do not supply independent pixel correspondence ground truth.
The available results establish
a detector reference and executable registration candidates, not a downstream
benefit from generated data or validation of the radiometric hypothesis.
An additional MS² source study tests calibrated RGB–thermal geometry without
substituting it for Anti-UAV300. Image matching and same-camera depth diagnostics
show remaining inconsistency; that source is also not qualified for paired
generator supervision.

## Method

### Controlled downstream comparison

The detector study separates three conditions: real imagery alone, real imagery
mixed with a deterministic unlearned pseudo-IR rendering, and real imagery mixed
with learned generated IR. The simulated condition is necessary to distinguish a
benefit specific to learned generation from a generic effect of extra non-real
images. Data ratios, training budget, detector initialization and evaluation
membership must be recorded for each completed comparison. The latter two
conditions do not yet have completed comparable detector results.

The completed reference uses YOLOX-s with six FLIR classes,300 epochs and the
frozen train/validation preparation and preprocessing. Its original manifest,
checkpoint, predictions and complete COCO metrics verify against their recorded
hashes. Two changed input source files are verified from the retained original
Git commit rather than asserted to match the current worktree. This summary's
export performs static verification, not a fresh GPU training replay.

### Registration candidate and control

The registration candidate freezes a previously trained shared-velocity predictor
and learns an image-conditioned residual. Its inputs are the RGB and IR images,
the base-warped RGB image, base velocity, source-support mask and coordinates.
Existing train boxes supervise the loss but do not enter prediction. A small
convolutional network predicts bounded spatial velocity controls; forward and
reverse fields are integrated from the combined velocity and its negative.

The control continues optimizing the existing predictor. Both conditions use
the same consumed training frames and3,200 optimizer updates over10 epochs.
Their learning rates differ:1e-5 for continuation and1e-4 for the residual head.
Consequently the comparison tests these complete training configurations, not
an isolated causal effect of architecture. The geometric loss and evaluation
predicates are unchanged.

The screen uses one fixed midpoint from each of160 official training sequences.
A pair passes only when both directions satisfy the box, centroid, area,
support, cycle and Jacobian conditions on that same pair. The aggregate threshold
is95% for both joint-frame and sequence-macro pass rates. These are training-set
engineering measurements. Small cycle error or box overlap alone does not prove
that pixels depict the same physical surface in both modalities.

### Additional-source geometry diagnostics

MS² is a separately authorized training-source candidate. Its camera calibration,
stereo pairs, timestamps, RGB odometry and projected single-scan LiDAR permit
checks that Anti-UAV300's supplied boxes alone do not provide. The study uses
fixed panels from one official training sequence; no independent-sequence
generalization or dense physical ground truth is claimed.

Three measurements must remain distinct: learned-match agreement with calibrated
projection, agreement between image stereo and sparse projected LiDAR, and actual
RGB–IR physical correspondence. Improvements in the first two do not establish
the third. Camera corrections fitted to learned matches remain exploratory, and
no corrected calibration has been installed as physical truth. Unsupported
reference points and unavailable exposure poses remain counted in full-reference
metrics rather than being removed to obtain a passing result.

## Results

### FLIR real-only detection reference

| Condition | Validation images | Epochs | AP@0.5 | AP@0.5:0.95 | AR@0.5 | AR@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Real-only YOLOX-s | 1,144 | 300 | 57.95 | 35.56 | 74.19 | 47.46 |
| Real + simulated | — | — | Not completed | Not completed | Not completed | Not completed |
| Real + generated | — | — | Not completed | Not completed | Not completed | Not completed |

The verified evaluation contains57,114 detections and excludes no boxes at the
nonfinite/nonpositive-extent filter. The following per-class values preserve the
reported COCO class mapping; the small truck AP remains visible rather than
being excluded from the aggregate.

| Class | AP@0.5 | AP@0.5:0.95 |
|---|---:|---:|
| Person | 80.80 | 44.32 |
| Bike | 46.82 | 28.08 |
| Car | 85.12 | 58.03 |
| Motor | 62.18 | 32.12 |
| Bus | 58.95 | 41.54 |
| Truck | 13.85 | 9.29 |

The reference run's technical ID is `flir_real_only_full_seed0_v2`. This denotes
the completed deterministic **detector baseline**, not the failed registration-v2
idea. Failed registration-v2 results are not used as the main comparator here.

### Anti-UAV300 registration engineering screen

| Configuration | Jointly passing pairs | Pass rate |
|---|---:|---:|
| Shared initialization before additional fitting | 114/160 | 71.250% |
| Existing-predictor continuation | 115/160 | 71.875% |
| Image-conditioned residual velocity | 117/160 | 73.125% |
| Required engineering level | At least152/160 | At least95% |

Relative to continuation, the residual configuration gains five passing pairs
and loses three, a net gain of two pairs or1.25 percentage points. This single-seed
train-screen difference does not establish generalization, statistical superiority
or physical registration accuracy. It is not a detector AP improvement.

The subsequent300-additional-epoch comparison is now complete and its saved
artifacts were verified on2026-09-24. Both arms begin from the same residual
checkpoint and retain96,000 optimizer updates under the same objective and
optimizer schedule; their sampled data differ by design.

| Seed0 additional-training condition | Midpoint passes | Quarter-frame passes |
|---|---:|---:|
| Uniform sequence sampling | 125/160 (78.125%) | 133/160 (83.125%) |
| Failure-aware sequence sampling | 124/160 (77.500%) | 133/160 (83.125%) |

There is no observed aggregate advantage for failure-aware sampling in this
single-seed run. Both remain below the95% train prerequisite. These are train
engineering measurements, not held-out accuracy or physical correspondence GT.
The [sampling experiment record](registration_replay.md) contains paired counts,
artifact hashes, resume verification and size-stratified failure attribution.
The earlier `results_snapshot.json` remains a frozen FLIR/residual10 snapshot;
it does not contain or verify these newly completed300-epoch results. Verify
the latter separately with `bash scripts/run_registration_replay.sh both compare`.

### MS² additional-source evaluation

These are diagnostic comparisons, **not detector AP or Anti-UAV300 accuracy**.
No table row constitutes qualification. Each linked record states its reference
population, support rules, source identities and complete controls.

| Diagnostic | Baseline | Candidate or finding | Interpretation |
|---|---:|---:|---|
| XoFTR match PCK3, 15-frame frozen-camera confirmation | 49.74% with its rotation-only correction | 57.60% with its augmented camera | Better agreement with fitted geometry, not independently measured physical accuracy |
| MINIMA match PCK3, same confirmation panel | 58.34% with its rotation-only correction | 68.11% with its augmented camera | Same limitation; no dense supervision approval |
| RGB/time-aware stereo versus projected LiDAR, 16-frame panel | — | 0.704 equivalent disparity px | Conditional median of frame medians; 52,801/98,628 points supported |
| Thermal/time-aware stereo versus projected LiDAR, same frame IDs | — | 1.094 equivalent disparity px | Conditional median of frame medians; 55,005/132,302 points supported |
| Thermal depth-encoding sensitivity | One encoded-depth-unit perturbation | At most 0.0632 equivalent disparity px | Too small to account for most observed stereo residuals under fixed geometry |
| RAFT-Stereo versus StereoSGBM, thermal/time-aware common points | 1.057 equivalent disparity px | 1.225 equivalent disparity px | Coverage increases, but common-point agreement worsens; not qualified |

The first two rows use the [15-frame confirmation panel](registration_ms2_intrinsic_confirmation.md).
The [LiDAR comparison](registration_ms2_lidar_stereo.md) uses a different,
pre-existing 16-frame panel, with two time-aware endpoint poses unavailable;
its conditional medians therefore represent 14 frames. RGB and thermal LiDAR
pixels are different reference populations, not matched identities of the same
physical returns. The [encoding sensitivity](registration_ms2_lidar_reference.md)
does not represent total LiDAR/calibration uncertainty. Low-local-depth-variation
thermal groups still retain approximately 1.05 equivalent pixels of discrepancy.
The [RAFT protocol](registration_ms2_raft_stereo.md) preserves the baseline's full
reference population and separates coverage from common-point accuracy.
Its completed 96-case GPU result passed CPU saved-array verification. Thermal
supported estimates increase from 55,005 to 102,980 of 132,302 reference points,
while the common-point comparison uses 53,065 points and improves the frame
median in only 4/14 available frames. The higher full-reference within-3-pixel
fraction (35.98% to 60.76%) must not be presented as improved common-point precision.

These MS² results are not included in the earlier `results_snapshot.json`.
Their linked reports and verification commands are the authoritative records.
All remain descriptive measurements from one training sequence, not a qualified
generator source or evidence for the radiometric downstream hypothesis.

## Conclusion

The completed evidence supports a usable real-only detector reference and a
small improvement in one registration training screen. It does **not** yet support
the claim that generated imagery improves detection, that RFS predicts held-out
detector gains, or that the paired generator's supervision is qualified.

Two different registration issues remain: meeting the existing geometric
coverage criteria, and obtaining independently supported physical correspondence
on the eventual supervision domain. More epochs address a learning hypothesis;
they do not create missing ground truth or implement the absent scientific
approval/export path. The three-arm detector comparison and its uncertainty/label
checks must be completed before presenting the proposed method as a validated
augmentation result. See the [completion audit](project_completion_audit.md) for
the remaining evidence requirements and the [sampling experiment](registration_replay.md)
for the current commands.

## Rechecking this summary

With the original local experiment artifacts and retained source commit available:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.export_public_results --check docs/results_snapshot.json
```

The checker verifies declared baseline artifacts and original source identities,
recomputes the completed registration comparison and compares the aggregate
snapshot exactly. It does not access validation/test pixels, run GPU inference,
submit jobs or publish to GitHub. Without the original artifacts, the public JSON
is a reported measurement record, not a standalone reproduction of training.
