# Frozen MS² effective-camera confirmation protocol

Status: 2026-09-24. **Extraction, CPU inference, saved-array verification and
image-structure checks are complete. Both own intrinsic augmentations improve
all15new frames relative to rotation-only. Registration and generator training
remain unqualified.** This follows the exploratory
[intrinsic sensitivity experiment](registration_ms2_residuals.md).

## Fixed candidates and fresh panel

All five cameras from the completed sensitivity report are retained: author
calibration, XoFTR-fitted rotation, MINIMA-fitted rotation, and each rotation
combined with its own four-parameter thermal-intrinsic correction. They are
copied into the plan before image extraction. No fitting, frame exclusion,
threshold changes or per-frame thermal normalization is allowed.

Source candidate report SHA256:
`d71ad7849fbc26bc1395b5475ed79978099bd148399a19a039df69c58b40e2ff`.

Each new frame is the integer midpoint between an original 16-panel left
endpoint and the midpoint previously used for the first confirmation. All 15
IDs are disjoint from the original 16 and previous 15:

`000174 000870 001566 002262 002958 003654 004350 005046 005742 006438 007134 007830 008526 009222 009918`

This remains the **same official training sequence**, not a different scene,
rig, official validation split or official test set. Temporal correlation and
unknown matcher pretraining overlap remain limitations.

Plan: `experiments/ms2_intrinsic_confirmation_plan_01/plan.json`, SHA256
`68dbed6ba590eef37b60d41f0c32d48971a18535dfbd8b7bfa0023556d467ce3`.

## Execution and denominator rules

The existing complete sync archive is read through its full CRC/EOF checks;
archive and extracted file hashes are verified. Partial files are not evidence
of complete extraction. The runner requires the completed extraction report
and rejects any change to the complete frozen plan or camera matrices.

Both frozen CPU matchers run on all 15 paired cases and all 15 cyclic-offset-7
wrong-thermal cases: 60 inference cases, each scored with all five cameras.
Native RGB stereo estimates depth; the original matcher input scaling,
half-pixel coordinate restoration and 3308–4974 DN thermal window are unchanged.
Ego-motion uses only the RGB trajectory, with **no extrapolation** allowed for
this interior-frame panel.

All candidate scores use the author-calibration reference support population.
Points that leave the image after correction remain failures. All raw match
counts and unsupported counts are retained alongside conditional statistics.
The existing 1/3/5/10-native-thermal-pixel scores are diagnostic metrics, not
new qualification thresholds.

The separate verifier checks every planned model/frame/condition/candidate,
hashes and saved match/stereo arrays. It resamples disparity, reconstructs depth,
projections and support, and recomputes scores. This verification is not another
neural inference run or independent physical-pixel ground truth.

## Commands and current state

Already completed: plan creation, full archive CRC/EOF verification and extraction.
**Do not launch a second extractor into the existing output directory.**
The completion report is:
`experiments/ms2_intrinsic_confirmation_sync_01_report.json`.
Its SHA256 is
`87dd2e1f23a31e48523313f59c91fe3c688fbf59c4d134f6dc6d8fb1ec654834`.
All117selected files are present; required-missing is empty.

The CPU inference and chained saved-array verification below have completed.
These are not GPU jobs. For later reproduction use fresh destinations rather
than the occupied paths shown below; no rerun is needed to finish these steps.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_intrinsic_confirmation infer \
  --out-dir experiments/ms2_intrinsic_confirmation_image_01

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_intrinsic_confirmation verify \
  --report experiments/ms2_intrinsic_confirmation_image_01/report.json \
  --out-dir experiments/ms2_intrinsic_confirmation_analysis_01
```

Fourteen added tests cover disjoint selection, frozen cameras, invalid fits,
changed intrinsics, fixed denominators, empty cases, complete saved-array
verification and rejection of altered scores, pairings, case sets, depths,
support masks and artifact bytes. Including the image-structure tests below,
the full CPU suite passed: **858 tests,
5 existing dependency warnings**. Repository Ruff and `git diff --check` pass.
No source calibration or generator authorization is changed.

## Image-structure corroboration

The two learned matchers share an architecture, so their agreement is not a
fully independent alignment reference. A separate CPU runner,
`scripts/probe_ms2_image_structure.py`, uses **no learned match coordinates**.
It derives a dense projection from native RGB stereo and each frozen camera,
warps thermal intensity into the native RGB grid, then computes signed gradient
correlation, absolute gradient correlation and squared gradient-vector cosine
similarity. Gradients are computed after warping, in the same RGB coordinates.
The thermal window and all five cameras remain unchanged; no score is optimized.

The scored mask is the intersection of all camera projection supports, eroded
by one pixel so every Sobel stencil stays supported. This is explicitly a
**common-subset appearance diagnostic**, not a fixed-full-image qualification
rate. Native pixel count, RGB-stereo support, each camera's support, eroded author
support and any additional common-mask loss are reported separately. No frame
is dropped; empty or textureless cases have missing scores, not perfect scores.
Wrong-thermal images use the same geometry and mask as their true-pair controls.

This offers image-based corroboration with a different representation, but still
shares the stereo/camera assumptions. Gradient agreement is not physical-pixel
ground truth. The added tests cover native half-pixel sampling, mask erosion,
identity/inversion/misalignment controls, and flat/empty inputs. Actual image
scores have been computed from the completed extraction, not inferred from unit tests.

The following CPU run has completed. Do not start a duplicate; later
reproduction requires a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_image_structure \
  --out-dir experiments/ms2_image_structure_01
```

## Completed new-frame confirmation

All60matching cases and all five cameras were retained. The verifier reconstructed
depth, support and scores and checked751dependency/artifact identities. The table
uses the **same author-calibration-supported point population** per model, and
equal-frame PCK3 over all15frames. These are native thermal pixel residuals
between calibrated stereo-depth projections and image matches, not physical GT.

| Evaluation model | Fixed camera | Supported / all paired matches | PCK3 | Median of frame median errors |
|---|---|---:|---:|---:|
| XoFTR640 | Author calibration | 13,271 /18,028 | 25.66% | 4.398px |
| XoFTR640 | Own rotation | same | 49.74% | 2.954px |
| XoFTR640 | Own rotation + intrinsic | same | 57.60% | 2.411px |
| XoFTR640 | MINIMA rotation + intrinsic | same | 57.42% | 2.458px |
| MINIMA-XoFTR | Author calibration | 11,076 /14,592 | 34.07% | 3.694px |
| MINIMA-XoFTR | Own rotation | same | 58.34% | 2.374px |
| MINIMA-XoFTR | Own rotation + intrinsic | same | 68.11% | 1.802px |
| MINIMA-XoFTR | XoFTR rotation + intrinsic | same | 67.37% | 1.865px |

Relative to **own rotation-only**, both augmented cameras improve15/15frames.
Relative to the **original author calibration**, they improve14/15frames and
worsen one frame for each matcher; this regression is retained. No camera loses
projected support. Own augmented per-frame PCK3 spans31.26–73.72%(XoFTR) and
36.51–84.48%(MINIMA), so the remaining errors are substantial.

Wrong-pair PCK3 is0.133%(XoFTR own augmented) and0.370%(MINIMA own augmented),
with median-of-frame errors89.12px/78.13px. Wrong-pair scores are not forced to
zero. The augmented cameras were not refitted or selected using this panel.
These results confirm transfer of a bounded effective-camera improvement to new
frames in the same training sequence, not to another scene, rig or official test.

The thermal window is unchanged; clipped fractions range0–5.51%, median0.53%.
All thermal poses are interpolated without extrapolation. Maximum absolute RGB
stereo timestamp skew is0.106707ms. Native RGB stereo support ranges238,077–337,208
of470,016pixels; it does not establish fully observed dense generator support.

Artifacts:

- Inference: `experiments/ms2_intrinsic_confirmation_image_01/report.json`, SHA256
  `e2b4e5ea6695204ee28a23349611de05ca4eecbbca5b815f46115136d4995d04`.
- Inference preflight SHA256:
  `488769ccbab5a9188d708b758e3ce87259c24a568830f38bb84417d0a0df67b2`.
- Recomputed analysis: `experiments/ms2_intrinsic_confirmation_analysis_01/report.json`, SHA256
  `15e4982a0dacfc3236b5577ca49a84447afd22f3c5dcd8bd0b55a2632300b5ef`.

## Completed image-structure results

All150frame/condition/camera score rows are present, with finite scores in all
cases. The common eroded mask contains177,490–289,326native RGB pixels per frame
(median244,189). In this panel there is **zero additional mask loss** relative to
the eroded author support. Unobserved pixels outside that mask remain unverified.

The table reports equal-frame mean similarities; larger is more similar. Signed
NGCC preserves contrast polarity, absolute NGCC discards global polarity, and
the squared-gradient-cosine diagnostic discards local polarity. None is a
calibrated correctness probability or a registration pass criterion.

| Condition / camera | Signed NGCC | Absolute NGCC | Squared-gradient cosine |
|---|---:|---:|---:|
| Paired / author | 0.00114 | 0.02299 | 0.45046 |
| Paired / XoFTR rotation | −0.01478 | 0.05830 | 0.46260 |
| Paired / MINIMA rotation | −0.01305 | 0.05916 | 0.46310 |
| Paired / XoFTR augmented | −0.03410 | 0.07296 | 0.46669 |
| Paired / MINIMA augmented | −0.03358 | 0.07131 | 0.46638 |
| Wrong pair / author | 0.00535 | 0.00975 | 0.41235 |
| Wrong pair / XoFTR augmented | 0.00432 | 0.01379 | 0.41272 |
| Wrong pair / MINIMA augmented | 0.00429 | 0.01279 | 0.41252 |

Both augmentations improve absolute NGCC over their own rotations on10/15frames
and squared-gradient cosine on14/15frames. Signed NGCC instead worsens on12/15
XoFTR and10/15MINIMA frames. This mixed result is retained: the absolute
correlations remain low and the squared-gradient gain is small. The measurements
provide limited representation-level corroboration, **not** strong independent
physical-alignment evidence. Different metrics are not cherry-picked into a pass.

Image-structure report: `experiments/ms2_image_structure_01/report.json`, SHA256
`8b62391253e2771ee0e7e9453f84766edcd219dc88c1530a2eb54a5f12748844`.
Its preflight SHA256:
`48af5d657a76eea04f1c5f24704aaabe2a36f75e7c50795e4b88041fda112c17`.
All678input/source identities were rechecked after completion.

## Decision

The focal/principal-point augmentation has a repeatable benefit relative to
rotation-only, but significant point disagreement, missing dense observation
support and uncertain physical calibration remain. No camera is installed into
the dataset, no generator export is authorized, and no Anti-UAV300 result is
relabelled. The next investigation must address those remaining limitations;
another training run or a success flag alone would not resolve them.

The subsequent [asynchronous thermal-stereo study](registration_ms2_cross_stereo.md)
uses the previously unused right thermal images. It finds substantial potential
motion-induced disparity error and mixed cross-depth evidence after correcting
for it; the effective camera is still not physically qualified.
