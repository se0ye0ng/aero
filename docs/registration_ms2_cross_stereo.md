# MS² asynchronous thermal stereo and cross-sensor depth check

Status: 2026-09-24. Timing analysis, time-aware rectification, actual stereo
experiments and repeat checks completed on the15quarter-interval frames.
**Registration and generator training remain unqualified.** No original camera
calibration, image, threshold or Anti-UAV300 result was changed.

## Why another stereo depth estimate needs timing correction

The previously used RGB stereo pair has sub-millisecond timestamp skew on this
panel. The thermal pair does not: recorded right-minus-left times range
−7.7320ms to+8.1198ms. Treating those exposures as simultaneous is an additional
motion hypothesis, not a property established by the dataset's stereo label.

The timing diagnostic uses a fixed16pixel-spaced native RGB grid, existing RGB
stereo depth, the supplied RGB odometry and the supplied timestamps. It projects
the same static3Dpoint into the thermal right camera twice: at the left exposure
time and at the recorded right exposure time. It uses neither learned RGB–IR
matches nor a newly fitted camera.

With the author camera, the median of frame-median displacements is0.296native
thermal pixels. Frame000870 has median1.729px, p952.674px and maximum6.036px.
Across the panel the largest frame-p95 vertical discrepancy is0.919px. On frame
000870, the p95 absolute horizontal motion shift is47.38% of the static stereo
disparity. Thus a small-looking exposure delay can matter for stereo depth.
These are model-based motion estimates, not measured shutter timing or a model
of independently moving vehicles.

Timing report: `experiments/ms2_thermal_stereo_timing_01/report.json`, SHA256
`783d0144f52b1976d547f9fe9ba8b1fc300a8f175312a5f0254860f72fda5f5d`.

This finding **does not invalidate the earlier RGB-stereo experiment** or prove
the cause of its RGB–IR residuals: that experiment did not use thermal stereo
depth. It prevents adopting an incorrectly assumed synchronous thermal reference.

## Implemented geometry and controls

For the time-aware variant, both thermal exposure poses are expressed through
the same RGB trajectory. The relative transform is
`T_right_at_right_time_from_RGB @ inverse(T_left_at_left_time_from_RGB)`.
The static control uses only the provided thermal stereo baseline. Corrections
are applied to the common left-camera frame before that physical baseline.

The rectification and disparity reconstruction use `stereoRectify`, remapping,
and its Q matrix as described in the [OpenCV calibration documentation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).
The actual installed OpenCV version is5.0.0; known3Dpoint tests check that API's
behavior, including explicit contiguous3×1translation input. Images are already
lens-undistorted, so no optical distortion is applied again. Rectification uses
alpha1 and equal virtual principal points. Remapping boundaries are masked with
the5×5stereo block footprint, rather than filled as valid observations.

The frozen StereoSGBM128-disparity/block5 settings, positive disparity, reverse
consistency≤1pixel and four-neighbor support rules are retained. The resulting
3Dpoints are rotated **back to the original thermal-left camera coordinates**
before comparing Z-depth. Virtual rectified depth is not confused with native Z.

All135cases are retained:15frames×3cameras×3conditions. Cameras are the author
calibration and the two previously frozen augmented cameras. Conditions are
paired/static, paired/time-aware, and unrelated right image/time-aware. The
wrong right image is the fixed cyclic-offset-7frame and uses true-pair geometry
deliberately as a negative control. No camera, stereo parameter or thermal DN
window is fitted using these scores.

For augmented-camera diagnostics, the same effective intrinsic correction is
applied to both thermal heads while retaining the author baseline. This is an
explicit extension of that camera hypothesis, **not independently verified
physical right-camera calibration**.

## Actual cross-depth results

The reference population is16,583author-geometry-supported grid points across
all15frames, fixed for every camera/condition. Missing thermal stereo estimates
remain failures in the full-reference score. Conditional statistics and a
separate static/time-aware common-support comparison are also recorded.

Relative depth error is `abs(thermal_Z - RGB_projected_Z) / RGB_projected_Z`.
The10% cut is a descriptive sensitivity readout, not a registration qualification
threshold. It is depth-dependent in disparity space; a fixed pixel error can
produce a large relative depth error at long range. Both depths are estimates,
and this comparison does not establish which estimate is correct.

| Camera | Thermal stereo | Available /16,583 | Equal-frame full-reference fraction within10% | Median of frame median relative errors on available points |
|---|---|---:|---:|---:|
| Author | Static | 10,370 | 18.61% | 18.68% |
| Author | Time-aware | 11,281 | 18.18% | 19.80% |
| XoFTR augmented | Static | 10,634 | 15.13% | 19.64% |
| XoFTR augmented | Time-aware | 11,380 | 16.10% | 20.44% |
| MINIMA augmented | Static | 10,641 | 15.32% | 19.97% |
| MINIMA augmented | Time-aware | 11,491 | 16.37% | 20.38% |

Wrong-right controls have only240–250available points across the same reference
population, with full-reference within10%fractions0.027–0.061%. Their conditional
median-of-frame errors are89.75–93.07%. This is consistent with meaningful
same-scene stereo signal, not a generally accurate independent depth reference.

On the **same points available in both static and timed variants**, median-of-frame
relative error changes as follows:

| Camera | Common points | Static | Time-aware | Frames with lower timed median error |
|---|---:|---:|---:|---:|
| Author | 9,953 | 18.84% | 19.10% | 8/15 |
| XoFTR augmented | 10,177 | 18.51% | 20.12% | 8/15 |
| MINIMA augmented | 10,236 | 19.86% | 19.18% | 8/15 |

Time-aware geometry increases supported depth coverage, but accuracy against the
other estimated depth source is mixed. The augmented cameras do not improve the
full-reference10%depth agreement over the author camera. Consequently, the
earlier improvement in2Dlearned-match residuals cannot be equated with validated
physical calibration. Remaining causes can include depth-estimation error,
occlusion, moving objects, exposure timing/trajectory error and camera modeling.
This experiment does not isolate a unique cause or license dropping bad points.

## Depth discrepancy expressed in fixed-scale disparity pixels

The saved estimates were also reconstructed in their per-exposure rectified
cameras and checked against every original depth score. The expected disparity
comes from the projected RGB depth; the observed disparity comes from thermal
stereo. Their difference is multiplied by `387.7869505 / rectified_focal_px` so
all camera/exposure variants use the same author thermal focal scale.
This is **equivalent stereo disparity discrepancy, not RGB-to-IR correspondence
error**. Two incorrectly associated locations on an equal-depth surface can
still agree. The one- and three-pixel readouts below are descriptive, not new
qualification thresholds.

All 16,583 original reference points remain in the denominator; missing thermal
depth is not converted to a successful match. Fractions are means of per-frame
fractions. Median discrepancies are medians of per-frame conditional medians.

| Camera | Stereo | Full-reference within 1 px | Full-reference within 3 px | Conditional median discrepancy (px) |
|---|---|---:|---:|---:|
| Author | Static | 25.93% | 59.10% | 1.191 |
| Author | Time-aware | 26.19% | 59.98% | 1.227 |
| XoFTR augmented | Static | 21.80% | 60.95% | 1.357 |
| XoFTR augmented | Time-aware | 23.83% | 60.40% | 1.308 |
| MINIMA augmented | Static | 22.16% | 61.08% | 1.351 |
| MINIMA augmented | Time-aware | 24.24% | 60.78% | 1.302 |

The wrong-right controls have conditional median discrepancies of 28.48–34.35
equivalent pixels and full-reference within-3-pixel fractions of 0.068–0.149%.
For the author/time-aware case, conditional median discrepancies in fixed source
RGB depth bins 0–10, 10–20, 20–40 and >=40 m are respectively 1.181, 1.201, 1.161
and 1.090 px. Thus the large relative-depth percentages should not be interpreted
as equally large correspondence-pixel errors. Nevertheless, discrepancies remain
at near and middle ranges too; this analysis does not attribute everything to
long-range depth sensitivity.

The same 10,250 points have thermal depth under all three time-aware cameras.
On that common subset, median-of-frame discrepancies are 1.172 px (author),
1.267 px (XoFTR augmented) and 1.239 px (MINIMA augmented). Each augmented camera
has a smaller per-frame median than the author camera on only 3/15 frames.
This coverage-controlled comparison **does not corroborate physically more
accurate calibration**, despite earlier learned-match improvements. It also
does not prove which camera or depth estimator is correct.

Analysis report: `experiments/ms2_cross_stereo_analysis_01/report.json`, SHA256
`2a5905c673f556b197a32c4adeea99fe038dfdac67efcdf0b5ebe62aeba92ed4`.
All 832 saved input/source/artifact identities were rechecked, and all 135
original depth scores and support masks were reconstructed. This is a
saved-array verification, not another image-stereo execution or physical GT.

## Evidence and reproducibility

- Current report: `experiments/ms2_cross_stereo_02/report.json`, SHA256
  `c237aa8b490943a222276f52cb662d5c0ab646dfcecec34bafa8d4250625d794`.
- Preflight SHA256:
  `ba2181b8a4fd0845413a90d0ac8f4991af7de5bb96dffd4674d3dc839c0247c2`.
- All694input/source identities and135saved array artifacts were rechecked.
- All135case rows, common-support results and array hashes match the initial
  `_01`execution. `_02`supersedes it only for a lint-only source line wrap;
  no algorithm, parameter or candidate changed between those executions.
- Full CPU suite: **872 passed, 5 existing dependency warnings**. Repository Ruff
  and `git diff --check` pass. Tests include known3Dreconstruction under rotation
  and translation, native versus virtual camera depth, known image disparity,
  remapping/search boundaries and missing-reference denominators.
  Disparity-analysis tests additionally cover inversion under camera rotation,
  depth-dependent relative errors, fixed depth strata and missing estimates.

No GPU was used. Optional reproduction, with fresh outputs:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_thermal_stereo_timing \
  --out experiments/ms2_thermal_stereo_timing_repeat01/report.json
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_cross_stereo \
  --out-dir experiments/ms2_cross_stereo_repeat01
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.analyze_ms2_cross_stereo \
  --out experiments/ms2_cross_stereo_analysis_repeat01/report.json
```

The analysis command above deliberately reads the frozen `_02` artifacts; it
does not automatically analyze the optional new image-stereo execution.

## Completed exploratory thermal preprocessing control

Both thermal heads originally used the same left-derived 3308–4974 DN window.
An additional experiment retains author/time-aware geometry, all 15 frames, the
same 16,583 reference points, the fixed stereo algorithm and wrong-right controls.
It changes only the thermal-to-uint8 preprocessing. Before running, the new
preflight freezes three modes: the original window; a shared window from the
minimum of the two image p1 values to the maximum of their p99 values; and each
image's own p1–p99 window. Percentiles use entire images, not selected matches or
depth scores. No camera is refitted. These are matching inputs, not calibrated
temperatures or modified generator targets.

All 90 cases completed (15 frames × 2 pairing conditions × 3 modes). Original
stereo arrays reproduce exactly for all 30 baseline cases, including negative
controls. Source and derived identities were checked: 727 inputs/source files
and 90 new array artifacts. The shared and independent windows were not tuned
after their scores were observed. Nevertheless, this is **exploratory reuse of
an already observed panel**, not fresh confirmation or qualification.

| Preprocessing, paired/time-aware | Available /16,583 | Full-reference within 1 px | Full-reference within 3 px | Conditional median discrepancy (px) |
|---|---:|---:|---:|---:|
| Original fixed window | 11,281 | 26.19% | 59.98% | 1.227 |
| Pair-shared p1–p99 bounds | 11,248 | 25.91% | 59.79% | 1.209 |
| Independent p1–p99 bounds | 12,697 | 27.79% | 66.56% | 1.235 |

On the same 10,664 points supported by all three preprocessing modes, conditional
median-of-frame discrepancies are respectively 1.173, 1.177 and 1.177 px.
Shared/independent preprocessing improves the per-frame median on 8/15 and 9/15
frames, respectively. There is no clear aggregate common-point accuracy gain.
The independent mode increases available estimates, not evidence that each
estimate is more accurate. Wrong-right controls remain low coverage (219–258
points) and full-reference within-3-pixel fractions remain 0.120–0.173%.

Original equal-frame mean clipped fractions are 1.02% for the left and 5.27% for
the right thermal camera. Shared windows reduce these to 0.92%/1.60%; independent
windows give 1.97%/1.98%. Different scene views also affect these distributions,
so this is not a measurement of sensor offset or proof of its causal role.
Preprocessing changes coverage, but does not resolve the remaining discrepancy.
No best-scoring mode is installed into the registration/generator pipeline.

- Report: `experiments/ms2_thermal_preprocessing_01/report.json`, SHA256
  `07fedf127c7836d03acec24ad30c10ae0434fd362e3121a55c44ec4a1075356d`.
- Preflight SHA256:
  `4a490dba65deadfab0b995f6ee0bdcfc0b2d5a985e15451d6c3e544e38abd78b`.
- Full CPU suite after this control: **878 passed**, five existing dependency
  warnings. Six new tests cover exact original preprocessing, known affine
  intensity changes, shared bounds, constant images and invalid input modes/types.

Optional CPU reproduction to a new directory:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_thermal_preprocessing \
  --out-dir experiments/ms2_thermal_preprocessing_repeat01
```

Neither a relative-depth percentage alone nor an increase in available estimates
is sufficient to authorize a dense generator export. The remaining work is to
separate stereo-estimation uncertainty from camera/temporal geometry error using
additional evidence; this comparison does not identify a unique root cause.

The [subsequent single-scan LiDAR comparison](registration_ms2_lidar_stereo.md)
has now completed on the original 16-frame panel. Same-camera thermal stereo
already disagrees with projected LiDAR depth by a conditional median of about
1.09 equivalent disparity pixels in the time-aware condition. This further
limits its use as a registration reference; it does not certify either source
or identify a unique calibration error.
