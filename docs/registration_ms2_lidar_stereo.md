# MS² same-camera LiDAR versus image-stereo depth

Status: 2026-09-24. The CPU experiment completed all 96 planned cases. It adds
an external range measurement to the earlier comparison between two image-depth
estimators. **Neither registration qualification nor generator training is
approved.** Anti-UAV300's failed gate remains unchanged.

## Measurement and controls

The official format identifies `depth` as a single-scan VLP16 projection into
each left camera, encoded as metric depth multiplied by 256. This experiment
does not use multi-frame or learned-filtered depth products.
[Official data-format description](https://sites.google.com/view/multi-spectral-stereo-dataset/data-format).

All original 16 training-panel frames are retained, with both RGB and thermal
stereo pairs. Each pair is evaluated with the author static baseline, the
timestamp-aware baseline, and an unrelated right image (fixed cyclic offset 8)
under the true-pair time-aware geometry. This gives 16 × 2 × 3 = 96 cases.
All camera parameters, StereoSGBM settings, reverse-consistency checks and
four-neighbor support rules are unchanged. Thermal inputs retain the original
3308–4974 DN display window; there is no percentile fitting in this experiment.

The reference consists of **every nonzero single-scan LiDAR depth pixel** in the
corresponding left-camera depth map: 98,628 RGB pixels and 132,302 thermal pixels
over the panel. Stereo is sampled at those exact native coordinates. Missing
stereo estimates remain failures in full-reference fractions. Zero LiDAR pixels
remain unmeasured; no depth filling creates additional reference points.

Both camera exposures use the same RGB odometry trajectory. For the left-camera
transform `F` from RGB and physical stereo baseline `B`, the right-from-left
transform is `B @ F @ inverse(P_right_time) @ P_left_time @ inverse(F)`.
There is no mixing of the independently zeroed sensor odometry origins. Poses
are interpolated only inside their recorded time range, with no extrapolation.
For both sensors, frames 000000 and 010441 require out-of-range exposure poses.
Their time-aware and wrong-right cases are explicitly unavailable, **not dropped**;
their LiDAR pixels still contribute to the full-reference denominator.

The camera-native Z estimates are compared with the supplied camera-native LiDAR
Z. Expected rectified disparity is also computed from the LiDAR depth and compared
with observed image disparity. Discrepancies are expressed at the common author
thermal focal length, 387.7869505 px, to avoid equating native RGB and thermal
pixel scales. The report also preserves native-pixel discrepancies and signed
errors. Fixed LiDAR depth bins are 0–10, 10–20, 20–40 and >=40 m.

The original RGB-only diagnostic did not apply the explicit remapping border
mask used by the newer rectification helper. These are newly executed cases,
not a claim of bitwise replay of that older RGB diagnostic.

## Results

Fractions are equal-frame means including unavailable frames as failures.
Discrepancy columns are medians of frame medians, conditional on available
stereo. Consequently the timed median represents 14 frames and the static median
16; the common-point comparison below controls that difference.

| Sensor | Stereo geometry | Supported / all LiDAR pixels | Full-reference within 1 equivalent px | Within 3 equivalent px | Conditional median absolute discrepancy | Conditional median signed discrepancy |
|---|---|---:|---:|---:|---:|---:|
| RGB | Static | 60,141 / 98,628 | 40.73% | 57.68% | 0.712 px | +0.049 px |
| RGB | Time-aware | 52,801 / 98,628 | 35.76% | 50.02% | 0.704 px | +0.042 px |
| Thermal | Static | 58,499 / 132,302 | 22.51% | 40.50% | 0.992 px | +0.622 px |
| Thermal | Time-aware | 55,005 / 132,302 | 18.25% | 35.98% | 1.094 px | +0.672 px |

The lower full-reference timed fractions cannot by themselves demonstrate worse
time compensation: they also retain the two unavailable endpoint frames. RGB
and thermal reference populations are different projected pixels, not a set of
identified common LiDAR returns. Their numerical comparison is therefore not a
controlled same-surface ranking of the two cameras.

On points supported in **both** static and time-aware variants:

| Sensor | Common points / frames | Static median discrepancy | Time-aware median discrepancy | Frames improved |
|---|---|---:|---:|---:|
| RGB | 50,752 / 14 | 0.693 px | 0.694 px | 7/14 |
| Thermal | 46,925 / 14 | 1.026 px | 1.076 px | 5/14 |

Time compensation does not give an aggregate improvement on this common subset.
The thermal signed discrepancy is predominantly positive in the aggregate:
estimated disparity exceeds the disparity predicted by LiDAR depth. This does
**not** establish a principal-point or baseline calibration error. Stereo
selection bias at small disparities, wrong surfaces, rasterization, motion and
the projected-depth calibration remain alternative explanations.

For thermal/time-aware stereo, depth-bin absolute discrepancies are 1.144,
1.299, 0.935 and 1.036 equivalent pixels, respectively. Signed discrepancies
are +0.615, +0.361, +0.771 and +1.027 px. The residual is not confined to distant
points. Conditional median relative Z-depth errors are 8.94% for RGB/time-aware
and 11.75% for thermal/time-aware; these are not dense accuracy guarantees.

Wrong-right controls retain only 1,354 RGB and 1,277 thermal supported estimates.
Their conditional median discrepancies are 19.16 and 29.00 equivalent pixels,
with full-reference within-3-pixel fractions of 0.068% and 0.089%. Real pairs
contain a meaningful stereo signal, but an available stereo estimate is not
automatically an accurate reference.

## Consequence for registration

The preceding cross-stereo disagreement cannot be assigned entirely to RGB–IR
camera misregistration: same-camera image stereo already disagrees with the
projected LiDAR measurement. In particular, thermal stereo is not established
as a sufficiently accurate independent depth reference for qualifying a dense
RGB-to-IR warp. The earlier effective-camera corrections still lack that
physical validation; they are not installed or used to authorize training.

The LiDAR projections themselves depend on calibration and have sparse sampling,
pixel rasterization, occlusion and acquisition-time uncertainty. Individual
return identities and motion-compensated per-return timestamps have not been
verified here. The comparison neither provides dense RGB–IR ground truth nor
isolates a unique error source. It does narrow the next question to the stereo
reference's systematic and support-dependent errors, before any camera refit
based on that reference would be defensible.

## Evidence and reproduction

A subsequent [depth-encoding and local-neighbourhood sensitivity analysis](registration_ms2_lidar_reference.md)
finds that one encoded depth unit cannot explain most residuals and that thermal
discrepancy remains around 1.05 equivalent pixels even in low-local-variation
groups. All reference points are retained; this is not a new approval criterion.

- Report: `experiments/ms2_lidar_stereo_01/report.json`, SHA256
  `a53e36e8b2fe7bfa91a9cc349d8e1b2c16d6415e1d434eee2ef6820536316282`.
- Preflight SHA256:
  `567ecd321baba762b1ed68e6eb5b6d26a71fe14d07522b4802453b7421ddc2a1`.
- All 291 input/source identities and 96 array artifact hashes were checked.
- Saved stereo maps were resampled to reconstruct all 96 depth/disparity arrays,
  depth-bin scores and aggregate scores. Planned case completeness and identical
  per-frame LiDAR reference coordinates/depths across conditions were verified.
  This is an array-level verification, not a second image-inference execution.
- Full CPU suite: **883 passed**, five existing dependency warnings. The five
  new tests cover synchronous/moving rig transforms, explicit no-extrapolation,
  unavailable-frame denominators and LiDAR depth strata.
- No GPU, camera fitting, frame selection or qualification-threshold change.

Optional CPU reproduction to a fresh directory:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_lidar_stereo \
  --out-dir experiments/ms2_lidar_stereo_repeat01
```
