# MS² LiDAR-reference sensitivity

Status: CPU analysis completed on 2026-09-25. This tests two possible explanations
for the [same-camera stereo residuals](registration_ms2_lidar_stereo.md), while
the [RAFT-Stereo GPU comparison](registration_ms2_raft_stereo.md) remains pending.
It does not change any frozen GPU input, camera parameter or qualification rule.

## Method

All 96 baseline cases and all nonzero single-scan LiDAR pixels are retained.
Saved stereo depth and disparity errors are resampled and checked against the
baseline. Original PNG coordinates/depths must exactly match the saved references.
Two unavailable endpoint poses per sensor remain unavailable, not extrapolated.

First, each measured depth is perturbed by **±1/256 m**, one encoded depth unit.
Expected disparity is recomputed through the unchanged per-exposure geometry.
The maximum absolute change at the two endpoints is reported in the same
equivalent thermal-disparity units as the baseline. This is a sensitivity test,
**not a total uncertainty bound** or an assumption about the provider's rounding
direction. LiDAR ranging, pixel rasterization, occlusion, motion and calibration
uncertainty are not represented by one depth unit. A nonpositive perturbed depth
is unavailable rather than clipped to manufacture a finite bound.

Second, LiDAR-only neighbourhoods partition every reference point. For native
square radii 2 and 4 pixels, count available returns including the centre. Fewer
than three returns is a separate sparse-neighbourhood group. Otherwise compute
the range of `f_thermal * stereo_baseline / depth` over measured neighbours and
report groups <=1 and >1 equivalent disparity pixel. These exploratory bins
are independent of the stereo residual, not qualification thresholds. Both radii
are reported; no best-radius selection, point removal or depth filling occurs.
The partition is fixed across static, timed and unrelated-right conditions.

Small local range is **not** a verified planar surface or absence of occlusion:
sparse returns can miss an edge. Native-pixel radii also differ in angular extent
between RGB and thermal, and the two sensors have different reference populations.
The strata therefore cannot establish a controlled RGB-versus-thermal ranking.

## Results

Time-aware stereo retains 52,801 RGB and 55,005 thermal estimates. Only 119 RGB
and 124 thermal supported residuals fall within the corresponding one-unit depth
perturbation. Maximum modeled changes over available reference geometry are
0.05877 and 0.06318 equivalent pixels, respectively. Frame-wise p95 changes range
from 0.01221–0.02049 px for RGB and 0.01824–0.02575 px for thermal. These are much
smaller than the baseline conditional frame-median discrepancies, 0.704 and
1.094 px. One encoded-depth-unit perturbation cannot account for most residuals.

The table shows thermal/time-aware results. Discrepancy is the median of available
frame medians (14 frames), not a pooled pixel median. Reference counts include
all 16 frames; supported counts retain the original stereo validity rules.

| Native radius | LiDAR-only neighbourhood group | Reference / supported | Conditional median discrepancy |
|---|---|---:|---:|
| 2 px | Fewer than 3 returns | 12,631 / 4,162 | 1.086 px |
| 2 px | Local disparity range <=1 px | 108,821 / 47,223 | 1.058 px |
| 2 px | Local disparity range >1 px | 10,850 / 3,620 | 1.400 px |
| 4 px | Fewer than 3 returns | 1,585 / 758 | 1.012 px |
| 4 px | Local disparity range <=1 px | 108,248 / 46,463 | 1.048 px |
| 4 px | Local disparity range >1 px | 22,469 / 7,784 | 1.290 px |

Higher local depth variation is associated with larger thermal residuals, but
substantial residuals remain in the low-variation group at both radii. This does
not identify calibration as the cause or justify masking all depth boundaries.
RGB's radius-2 group is mostly sparse (95,933/98,628 points), illustrating why
unmeasured neighbourhood structure must not be called a smooth surface.
The machine-readable report includes all sensors, conditions, counts,
full-reference fractions, signed errors and both radii, including adverse controls.

## Consequence

Do not repair the discrepancy by changing the depth encoding, removing difficult
pixels or applying an assumed constant camera correction. These results narrow
two explanations but do not isolate a unique cause. The unchanged, same-reference
RAFT comparison remains the next estimator test. Its outcome must still separate
coverage from common-point accuracy; improvement alone will not qualify dense
RGB–IR supervision or replace the failed Anti-UAV300 result.

## Evidence and reproduction

- Report: `experiments/ms2_lidar_reference_sensitivity_01/report.json`.
- SHA256: `03bc5f2cc9b91f38b3bb55c1536517bf13dd36ee8f51f85c68efa4b54fb0a03e`.
- 390 input/source/artifact identities checked before completion.
- All 96 baseline depth/disparity arrays and depth-stratum scores reconstructed.
- Eight focused tests cover exhaustive strata, sparse/border/empty support,
  analytic inverse-depth sensitivity, unavailable perturbations and invalid settings.
- A separate CPU process reproduced the report byte-for-byte at
  `experiments/ms2_lidar_reference_sensitivity_repeat01/report.json`; all 192
  case/radius partitions preserve both reference and supported counts.
- Full CPU suite: **912 passed**, five existing dependency warnings. Ruff and
  `git diff --check` passed. All 421 frozen RAFT input identities remain unchanged.
- No GPU, new model, camera fit, inference rerun or generator approval.

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/aero-ms2-mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.analyze_ms2_lidar_reference \
  --out experiments/ms2_lidar_reference_sensitivity_new/report.json
```
