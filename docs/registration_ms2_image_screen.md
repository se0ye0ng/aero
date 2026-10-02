# MS² image-only matching versus calibrated sparse references

Status: the96-case CPU image screen completed on2026-09-24. The calibrated
reference scores do **not** establish registration qualification. Additional
known-resize, depth-free epipolar and same-modality stereo diagnostics also
completed. None is a registration/generator approval.
This is a training-only diagnostic for an explicitly authorized additional source,
not an Anti-UAV result or a replacement for its failed qualification.

## Frozen inputs and comparison

The same16frame IDs from `_2021-08-06-10-59-33` are retained. The completed native
audit and ego-motion diagnostic are verified by hash before use; see
[source acquisition and geometry evidence](registration_ms2_source.md).
Configuration: `configs/experiment/registration_ms2_image_cpu.yaml`.

Two existing checkpoints are tested without fine-tuning: original XoFTR640 and
MINIMA-XoFTR. The upstream XoFTR code is fixed at
`e0fbea431b30be9742effbf5577c90aa8eb938f9`; checkpoint identities are in the config.
Only preexisting untracked bytecode caches are tolerated in that vendor checkout,
and an isolated bytecode location is used for this run. All source/weight hashes
are rechecked after inference.

Per model and frame, run RGB→thermal and thermal→RGB on the correct pair, plus
RGB→thermal with the thermal frame shifted cyclically by8positions in the frozen
16-frame list. The unrelated condition is deliberately evaluated against the
correct-pair reference to expose apparent success without the correct target
image. It is not a valid physical pair. This gives96case rows across both models.
A known(+8,-8) input-pixel shift of the first RGB image is a separate execution
control for each model, not cross-modal accuracy evidence.

RGB is converted to grayscale. A single thermal display/matching window is fit
at the pooled training-panel0.5th/99.5th percentiles:3308–4974DN. It is fixed before
the first inference and used for every frame and both models. Inputs are bounded
at640pixels on the longer side with8-pixel divisibility, and match centres are
restored with the actual x/y scales and half-pixel convention. Native thermal
counts, native depth, RFS targets and generator targets are not modified.

Only image matches/confidences enter local-consensus TPS fitting. The existing
FPS policy and144-control limit are reused without fitting to MS² reference
scores. The fitted map predicts only inside its control convex hull and image
bounds; unsupported predictions count in the reference-score denominator.

## Reference scope and limitations

Sparse references project observed source depth through supplied calibration,
using either static geometry or the previously inspected RGB-trajectory
ego-motion transform. Both variants are prepared before image inference. The
bounded ego variant records two endpoint extrapolations, not16/16 strict
interpolations. Reference query locations do not enter image matching, control
selection or TPS fitting.

All three masks are reported separately:

1. Geometrically supported source-depth projections into the target image.
2. Nearest-centre z-buffer candidates among these sparse projected samples.
3. The z-buffer subset having a target nonzero-depth pixel within1native pixel
   and0.03m depth discrepancy.

The third mask is a pre-inference **internal-consistency slice**, not an accuracy
gate. No depth hole is filled. An unobserved occluder can invalidate even a
z-buffer candidate; nearest depth pixels are not confirmed identical LiDAR
returns. Raw source-point counts and reference counts accompany every slice.
Independent object motion, rolling shutter, calibration uncertainty and exact
exposure timing remain unresolved. The depth products can share calibration.

Report1/3/5/10target-native-pixel error fractions and conditional finite error
quantiles. Reverse-direction RGB pixels have a different angular scale from
thermal pixels. These are **calibrated-reference consistency** scores, not
independently verified physical pixel GT, not held-out model performance, and not
dense generator-training approval. No qualification threshold is created from
these observed scores.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_image_matching \
  --out-dir experiments/ms2_image_matching_repeat01
```

Use a fresh output directory. The completed original is
`experiments/ms2_image_matching_01/`. Its `preflight.json` was written before
inference, with SHA256
`73a89f7781c0f17325c6d11ee42ae3da61491bd7db1a331c82796c91c66ecee2`.
It records the window, reference files, pose interpolation details and verified
dependencies. Native correspondence arrays and complete case scores are saved;
no raw imagery or third-party weights are added to Git.

## Completed image-screen results

The primary table below uses the ego-motion reference and its pre-inference
depth-consistency mask. PCK3 is the fraction within3target-native pixels, averaged
equally over the16frames. Unsupported TPS predictions remain in the denominator.
The median is the median of frame medians **conditional on finite predictions**;
it must not be read as an all-reference error. RGB and thermal pixels have
different angular scales, so forward/reverse3-pixel thresholds are not equivalent.

| Frozen matcher | Direction / condition | References | Unsupported | Conditional median (target px) | Equal-frame PCK3 |
|---|---|---:|---:|---:|---:|
| XoFTR640 | RGB→thermal, paired | 72,245 | 17,980 | 5.271 | 18.47% |
| MINIMA-XoFTR | RGB→thermal, paired | 72,245 | 20,379 | 4.300 | 18.81% |
| XoFTR640 | Thermal→RGB, paired | 38,003 | 5,024 | 9.227 | 4.52% |
| MINIMA-XoFTR | Thermal→RGB, paired | 38,003 | 8,297 | 7.616 | 5.78% |
| XoFTR640 | RGB→unrelated thermal | 72,245 | 63,598 | 37.983 | 0.160% |
| MINIMA-XoFTR | RGB→unrelated thermal | 72,245 | 71,369 | 26.172 | 0.036% |

The broader ego-motion geometric mask gives forward PCK3 of16.87% and19.57%,
respectively. Therefore the low consistency is not confined to the stricter
depth-consistency slice. Static and ego-motion depth-consistency masks contain
different reference populations; their conditional metrics are not a paired
comparison on identical pixels.

The(+8,-8) same-image shift control returned1,500matches per model and median
errors of0.0524/0.0526input pixels. These controls test execution, not cross-modal
physical accuracy. The correct pairs are substantially better than unrelated
pairs, but neither pretraining nor shared-depth-product agreement proves that
the remaining several-pixel disagreement is caused by the matcher alone.

Artifacts:

- Image report: `experiments/ms2_image_matching_01/report.json`, SHA256
  `3abbeca131b191c85c24357ea6758c0acae9dcbdd51fffd5a066e3718d717d39`.
- Verified analysis: `experiments/ms2_image_matching_analysis_01/report.json`,
  SHA256 `f6de3ef060d07378572c6bd342d54b8a17730ac22f6db9e3e29c574fc8acb6ea`.
- Run the analysis with `python -m scripts.analyze_ms2_image_matching --report
  experiments/ms2_image_matching_01/report.json --out <fresh-report-path>`.

## Post-screen coordinate and depth-free controls

These diagnostics were specified **after** the image-screen results, not presented
as part of the original frozen protocol. The same16training frames and two frozen
models are retained. No acceptance threshold, image pairing, camera calibration,
depth value, matcher weight or generator target was changed.

### Known cross-resolution correspondence

Resize each actual RGB image(1224×384) to a synthetic thermal-shaped640×256
canvas using OpenCV area resampling. Process the original through the existing
640-long-side adapter, which produces640×200. Match both directions and restore
native coordinates with the existing half-pixel convention. Exact resize
correspondence is known analytically; no matcher fit supplies the reference.
This is a same-modality synthetic control, not real RGB–IR GT.

| Matcher | Target coordinate system | Matches over16frames | Median of frame median errors | Equal-frame fraction within3px |
|---|---|---:|---:|---:|
| XoFTR640 | Synthetic640×256 canvas | 32,159 | 0.246px | 99.91% |
| MINIMA-XoFTR | Synthetic640×256 canvas | 31,925 | 0.295px | 99.88% |
| XoFTR640 | Native1224×384 RGB | 32,056 | 0.385px | 99.77% |
| MINIMA-XoFTR | Native1224×384 RGB | 31,882 | 0.468px | 99.71% |

This makes a several-pixel resize/restoration bug unlikely on the tested images.
It does **not** establish correct camera calibration or multimodal localization.
Report: `experiments/ms2_resize_control_01/report.json`, SHA256
`e67ccec61a0001694d89209ac1bec4129058d958c3c8ce5b45a07126b9094c26`.
Reproduce with `python -m scripts.probe_ms2_resize_control --out-dir <fresh-dir>`;
the script is CPU-only.

### Depth-free epipolar geometry

Saved raw image matches are compared to epipolar lines defined by the supplied
intrinsics/extrinsics and the same bounded ego-motion model, without using depth
values. All returned matches enter each frame's fraction. This is not TPS scoring
and uses a different denominator from the main table.

| Matcher | RGB→thermal condition | Median of frame median line distances | Equal-frame fraction within3thermal px of line |
|---|---|---:|---:|
| XoFTR640 | Paired | 1.647px | 62.41% |
| MINIMA-XoFTR | Paired | 1.348px | 72.04% |
| XoFTR640 | Unrelated thermal | 69.529px | 2.71% |
| MINIMA-XoFTR | Unrelated thermal | 67.842px | 2.33% |

Points near the correct epipolar line may still be far from the correct position
**along** that line. Hence this better-looking result does not override the full
coordinate discrepancy. It shows useful image-pair structure but does not isolate
calibration error, depth projection error, motion error or matching error.

Report: `experiments/ms2_epipolar_02/report.json`, SHA256
`9a420c96663c93e4d6b95e5938bd28c0e8e46973f2b0daad86c6063cffced423`.
`_01` is superseded only because two source lines were shortened to satisfy lint;
no calculation changed. Reproduce with `python -m scripts.analyze_ms2_epipolar
--out <fresh-report-path>`.

### Same-modality stereo diagnostic

This completed64-case diagnostic uses the already extracted left/right RGB and left/right
thermal images, without acquiring or reading held-out frames. It reuses both
frozen matchers, the original thermal display window, native pixel coordinates,
and supplied stereo baselines. Both static and time-compensated geometry are
reported; left/right timestamp differences are retained. Unsupported pose
interpolation is reported rather than extending the20ms extrapolation bound.
Depth-projection checks use only nearest source-depth anchors within0.5native
pixels and remain approximations, not verified subpixel image GT.

All32sensor/frame combinations had a valid bounded motion transform; both models
were scored on all of them. Results below use the ego-motion variant. Medians are
medians of conditional frame medians; fractions are equal-frame averages.

| Matcher | Stereo modality | Image matches | Median distance to right epipolar line | Within3px of line | Matches near source depth | Median full depth-projection error | Within3px of depth projection |
|---|---|---:|---:|---:|---:|---:|---:|
| XoFTR640 | RGB | 26,866 | 0.411px | 99.75% | 271 | 1.859px | 72.29% |
| MINIMA-XoFTR | RGB | 27,037 | 0.440px | 99.80% | 267 | 1.697px | 74.36% |
| XoFTR640 | Thermal | 35,026 | 0.548px | 98.88% | 1,457 | 1.885px | 74.14% |
| MINIMA-XoFTR | Thermal | 34,120 | 0.558px | 99.23% | 1,319 | 1.710px | 75.48% |

Same-modality line agreement is much stronger than cross-modal line agreement,
and the known-resize error is small. Nevertheless, depth-based point prediction
already has nonzero residual within a single modality. This rules out treating
internally consistent projected depth as exact pixel GT. It does not uniquely
identify whether cross-modal residuals arise from extrinsics, motion, image
matching or depth-to-image registration. No rectification/calibration value was
silently modified to obtain these results.

Report: `experiments/ms2_stereo_control_01/report.json`, SHA256
`07b2ca32ad41c48effcb39379976b691f9d2cee49103bdce480bc251d530fe69`.
All223dependency hashes,64match files and the preflight hash were rechecked after
the run. The full CPU suite passed813tests with five existing warnings before
the subsequent calibration-candidate diagnostic was added.

Command: `python -m scripts.probe_ms2_stereo_control --out-dir <fresh-dir>`.
No new GPU training is justified by these diagnostic results alone. Anti-UAV's
failed qualification and the generator-training HOLD remain unchanged.

## Fixed-rig correction candidate: tested, not adopted

The next CPU diagnostic asks whether a **single** small correction to the
RGB-to-thermal camera transform explains the additional cross-modal error. It
does not fit per-frame flow, replace source files, tune timestamp offsets or
change intrinsics. This is an exploratory calibration-repair test, not a claim
that the author calibration is wrong.

The previously inspected16training frames are divided by their frozen-list index:
even indices for fitting, odd indices for checking. The checking frames do not
enter parameter fitting, but they are **not an unseen official test set** because
the initial image screen already used them. Two candidates per matcher are fit:
rotation only and a6-parameter rigid correction. Each rotation-vector component
is bounded to±3degrees; each translation component to±0.1m. These are diagnostic
search bounds, not physical uncertainty estimates or acceptance thresholds.
Optimization uses a1thermal-pixel-scale soft-L1 residual with equal per-match
weight and at most1,000function evaluations. No check-frame score selects a
candidate or extends its bounds.

Source RGB matches are associated with a measured depth pixel within0.5native
pixels, backprojected at the actual match position, and transformed through the
declared ego-motion geometry. All such associations are retained; the optional
cross-modal depth-consistency slice is **not** used to select fitting points.
Consequently these rows use a different population from the main TPS table.
Each fitted correction is also scored on the other matcher's saved points.

| Evaluation matcher on8check frames | Author calibration PCK3 | Own rotation-only correction PCK3 | Own rigid correction PCK3 | Author / rotation / rigid median error |
|---|---:|---:|---:|---:|
| XoFTR640 | 34.29% | 36.58% | 31.50% | 4.185 /3.518 /3.918px |
| MINIMA-XoFTR | 47.39% | 47.48% | 41.23% | 3.372 /2.872 /3.416px |

Fractions are equal-frame averages; errors are medians of frame medians.
Every check frame has associated matches(108XoFTR,58MINIMA in total), but these
are sparse nearest-depth approximations, not exact pixel GT. Both6-parameter fits
reach the+0.1m target-Z search boundary; the MINIMA fit also reaches the−0.1m
target-X boundary. Their full numerical Jacobian ranks do not prove physical
identifiability or an accurate calibration. A lower fit residual and optimizer
success are insufficient: neither rigid candidate improves its own check PCK3.
**No correction is adopted, and the source calibration stays unchanged.**

The bounded experiment does not prove all recalibration impossible. It rejects
the tested single-rig correction as a demonstrated repair and prevents promoting
a visually plausible constant offset without check-frame evidence. Remaining
alternatives must explicitly distinguish local depth association/occlusion,
motion and true cross-modal matching error; simply broadening the search until
the same check data looks better would not establish qualification.

Report: `experiments/ms2_calibration_refinement_01/report.json`, SHA256
`819190da55ab2212f7d9916df79bc3631f1df68baf9f0ae9fee90bf840289204`.
The preflight records the8/8frame split and all bounds before optimization.
Command: `python -m scripts.probe_ms2_calibration_refinement --out-dir <fresh-dir>`.
Synthetic known-rigid recovery and behind-camera rejection tests pass.
Final repository verification after this diagnostic:815tests passed(five existing
dependency/API warnings), Ruff passed, and `git diff --check` passed. All253
calibration-diagnostic dependency hashes and the preflight hash were verified.
These software/integrity checks do not change the scientific qualification state.

A subsequent [RGB-only stereo-depth comparison](registration_ms2_stereo_depth.md)
tests whether the sparse depth associations hid a repeatable calibration error.
Its higher-support rotation-only fits improve exploratory check-frame scores;
the two frozen candidates also improve all15new confirmation frames, but are not
adopted as qualified calibration. This does not retroactively change the unsuccessful LiDAR-supported
fit above or make either depth source independent pixel GT.
