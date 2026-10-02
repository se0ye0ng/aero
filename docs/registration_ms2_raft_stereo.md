# MS² pretrained stereo-estimator comparison

Status: **all 96 GPU cases completed and saved-array verification passed** on
2026-09-25. Coverage increases, but common-point LiDAR discrepancy worsens in
the aggregate. Registration and generator training remain unqualified.
This is an inference-only
experiment, not another registration training run, camera calibration fit or
generator-training authorization.

## Why this experiment

The [LiDAR/stereo comparison](registration_ms2_lidar_stereo.md) found residual
disagreement between classical image stereo and single-scan projected LiDAR.
Consequently thermal StereoSGBM depth cannot simply be promoted to an accurate
registration reference. This experiment tests a different pretrained estimator
while retaining the same geometry and measured-reference population.

The [official RAFT-Stereo repository](https://github.com/princeton-vl/RAFT-Stereo)
provides a Middlebury checkpoint recommended by its authors for in-the-wild
images, and a regular PyTorch correlation path. We use that fixed checkpoint
without fine-tuning or compiling optional CUDA correlation extensions. This is
not a claim that RAFT-Stereo is the latest/best model or that it generalizes
successfully to thermal imagery; the comparison must measure that behavior.

Reviewed source commit: `6e93ed2169bd858dbb43033988563f3b0bb49506`.
Official linked checkpoint: Google Drive ID `1m3KoukUmKDoMv-ySOO6vBzYfWLyj9yqd`.
Observed checkpoint SHA256:
`d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819`.
This is a locally observed hash of the official download, not an independently
published checksum. The loader verifies it and requires strict parameter loading
with `torch.load(..., weights_only=True)`. No permissive/unrestricted pickle
fallback or random-weight fallback is provided.

## Frozen comparison

- All 96 baseline cases: 16 frames × RGB/thermal × static/timed/wrong-right.
- The same author camera matrices, exposure poses, exact LiDAR pixel coordinates
  and nonzero LiDAR depth values. Unavailable endpoint poses remain unavailable
  cases in the full-reference denominator; no extrapolation or frame exclusion.
- Native image dimensions, 32 recurrent updates, float32, regular correlation,
  no fine-tuning, no camera fit and no output-error-dependent filtering.
- Native RGB color; thermal 3308–4974 DN window replicated into three channels.
  RGB color differs from the baseline's grayscale representation. This is an
  explicitly labeled estimator-and-input comparison, not a pure algorithm-only
  ablation.
- Symmetric padding to multiples of 32 is removed before sampling. RAFT's output
  horizontal flow is negated to obtain positive left-minus-right disparity.
- Separate mirrored/swapped inference provides reverse disparity. Positive
  disparity below 128 pixels, reverse agreement <=1 pixel, the baseline remapping
  footprint and the existing four-neighbor support rules are retained. These
  support checks do not establish physical correctness.
- Report full-reference scores, conditional scores, depth strata and scores on
  the same points supported by both estimators. Preserve wrong-right cases and
  all failures. Equivalent-disparity units are not RGB–IR correspondence errors.

Before real inference, a fixed known 8-pixel synthetic shift must pass the
execution check. The check cannot qualify real registration. The bash wrapper
then runs a separate CPU reconstruction of the saved scores; this is not a
second GPU inference replay.

## Completed local validation

The actual official checkpoint loaded strictly on CPU. A 97×163 synthetic pair
exercises padding and unpadding; all 7,475 prespecified interior points were
within 1 pixel of the known disparity. Median error was 0.03622 px, p95 0.09658 px.

- Control: `experiments/ms2_raft_stereo_cpu_smoke_01/control.json`, SHA256
  `b40b6605ac8ed8665985023d73a212ab32695c6aaacf1ddf0761af6c78aa32f1`.
- Preflight SHA256:
  `6ba54de0efb331a1ba0418684038cbb503ac166d724e8ef29c0ef691a7f249c1`.
- Full-experiment preflight checked 96 cases and 421 input/source identities.
- Full CPU suite: **891 passed**, five existing dependency warnings. Ruff,
  shell syntax and `git diff --check` passed.
- Upstream AMP and meshgrid deprecation warnings also occur in real-model smoke
  inference; they did not prevent the check. No GPU was used locally.

Only `opt_einsum==3.4.0` was added to the existing virtual environment. PyTorch
and CUDA were not replaced. The optional dependency is recorded separately in
`requirements/raft-stereo-runtime.txt`; no downloaded source or weights are
included in Git.

## GPU command

On the allocated GPU node, in this shared checkout:

```bash
bash scripts/run_ms2_raft_stereo.sh
```

The wrapper preserves scheduler `CUDA_VISIBLE_DEVICES`, temporarily removes
`LD_LIBRARY_PATH` for the launched commands, runs CPU regression/preflight checks,
then GPU inference and CPU verification. There is no automatic CPU fallback,
downscaling, mixed-precision fallback or parameter change if GPU inference fails.

Default result: `experiments/ms2_raft_stereo_01/report.json`.
Existing output directories are refused. For an intentional new run:

```bash
AERO_MS2_RAFT_OUTPUT="$PWD/experiments/ms2_raft_stereo_repeat01" \
  bash scripts/run_ms2_raft_stereo.sh
```

Do not edit recorded source/input files during the run. A later source hash
mismatch is not permission to rewrite old manifests. If a different virtual
environment is required, install the optional dependency there first and set
`AERO_PYTHON` to that environment's Python executable. The downloaded vendor,
checkpoint and MS² artifacts must also be present and match the frozen hashes.

Improved sparse LiDAR agreement would support a further geometry check, not
automatically approve dense paired generator supervision. Pretraining exposure,
domain mismatch, shared calibration and sparse reference coverage still limit
the scientific interpretation. Anti-UAV300 qualification is not replaced.

## Prepared post-run comparison (CPU only)

`scripts/analyze_ms2_raft_stereo.py` is separate from the frozen inference runner;
adding it does not change the 421 identities recorded by the existing GPU plan.
Those identities were rechecked after implementation and remain unchanged.

After the complete GPU report exists, run:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.analyze_ms2_raft_stereo \
  --run-dir experiments/ms2_raft_stereo_01 \
  --out experiments/ms2_raft_stereo_comparison_01/report.json
```

The analyzer refuses absent reports and does not turn partial inference into an
analysis result. It validates the frozen case list, right-image pairing, geometry
availability, model identity, inference settings and non-qualification flags.
It additionally rebuilds remapping support and reverse-consistency validity from
saved forward/reverse disparity and the original geometry. A reported valid mask
is not simply trusted. Both estimators' depth/disparity scores are reconstructed.

For every sensor, pairing condition and fixed LiDAR depth stratum, the output
separates full-reference accuracy, estimated-point coverage, conditional accuracy
and accuracy on the exact same supported points. Frame improvements, regressions,
ties and unavailable common support are reported. These are descriptive statistics
on one training sequence, not independent-sequence confidence intervals or a new
qualification rule. The synthetic control remains only an execution check.

Thirteen additional tests cover changed protocols/pairings, duplicate cases,
unsupported qualification claims, altered validity masks, unavailable poses,
coverage-versus-accuracy separation and missing-result handling. Full CPU suite:
**904 passed**, five existing dependency warnings; Ruff and diff checks pass.
At preparation time the default GPU output directory was absent. The subsequent
completed GPU result is recorded below; the original inputs were not changed.

## Completed GPU result

The user ran inference on node31, using the NVIDIA GeForce RTX 4090 with
PyTorch 2.9.0+cu128 and CUDA runtime 12.8. All 96 planned cases are present.
The GPU synthetic shift control has all 7,475 points within 1 px, with median
absolute error 0.03621 px and p95 0.09659 px. This remains an execution control,
not evidence of real physical correspondence.

The separate CPU analyzer passed score reconstruction, case/protocol checks,
original baseline resampling, remapping-mask reconstruction, reverse-consistency
reconstruction and 521 input/source/artifact identity checks. No GPU inference
replay was performed during verification.

Time-aware cases are summarized below. Full-reference fractions are equal-frame
means over all 16 frames, retaining the two unavailable endpoint poses as failures.
Common-point discrepancies are medians of frame medians on 14 frames; the exact
same points are used for both methods. Units are equivalent thermal disparity
pixels, **not RGB–IR correspondence error**.

| Sensor | Method | Supported / reference | Full-reference within 3 px | Common-point median discrepancy |
|---|---|---:|---:|---:|
| RGB | StereoSGBM | 52,801 / 98,628 | 50.02% | 0.683 px |
| RGB | RAFT-Stereo | 77,834 / 98,628 | 74.76% | 0.734 px |
| Thermal | StereoSGBM | 55,005 / 132,302 | 35.98% | 1.057 px |
| Thermal | RAFT-Stereo | 102,980 / 132,302 | 60.76% | 1.225 px |

Common support contains 50,135 RGB and 53,065 thermal points. RAFT improves the
common-point frame median in only 3/14 RGB and 4/14 thermal frames; the other
11/14 and 10/14 worsen, with no ties. Its conditional discrepancy on its own
larger support is 0.915 px RGB and 1.541 px thermal, versus 0.704/1.094 px for
StereoSGBM. Coverage and accuracy therefore give different conclusions.

Static cases show the same trade-off: common-point medians are 0.694 to 0.750 px
for RGB and 0.947 to 1.096 px for thermal (baseline to RAFT), with improvements
in only 3/16 and 4/16 frames. RAFT wrong-right controls retain no RGB estimates
and 133 thermal estimates across three frames; none fall within 3 equivalent px
of the LiDAR reference. This distinguishes real-pair signal from unrelated inputs
but does not certify every real-pair estimate.

**Decision:** do not promote RAFT disparity to an accurate dense registration
reference. It increases usable coverage under the existing support rules but
does not improve common-point agreement with projected LiDAR. The reference itself
shares calibration and projection uncertainty, so the result cannot uniquely
attribute error to either estimator or calibration. No constant offset, camera
refit, failed-frame removal or qualification relaxation is adopted. The next
attribution should distinguish image-supported disparity from LiDAR-projection
disagreement using the saved maps before requesting another training run.
Anti-UAV qualification and generator authorization remain unchanged.

### Completed artifacts

- GPU report: `experiments/ms2_raft_stereo_01/report.json`, SHA256
  `ed2122fb173ad1d8e8c4df9f6c3464a1805fb270b5465e76bfe6331276334800`.
- GPU preflight SHA256:
  `54359f0d9998e9951b939769fa995f8c325dc2e3725c227ef3181a9f7287a73c`.
- CPU comparison: `experiments/ms2_raft_stereo_comparison_01/report.json`, SHA256
  `6634543db472ab2574bfcda03f10688c3858ddcf5442873d9a954c88fca3266a`.
- Comparison includes both sensors, all conditions and every fixed depth stratum;
  the table does not replace the complete recorded controls or missing support.

## Image patch evidence

A CPU follow-up completed on 2026-10-01 compares the three fixed disparity
predictions with the actual left/right images. Both image estimators give stronger
local patch agreement than LiDAR-predicted disparity, but incorrect matches in
the unrelated-image control can also score highly. Thus neither LiDAR agreement
nor local photometric agreement alone supplies a physical correspondence gate.

The diagnostic retains all 96 cases and every original LiDAR reference point.
It rectifies the same native grayscale RGB or fixed-window thermal images through
the unchanged exposure geometry, then samples 9 by 9 patches using float64
bilinear interpolation. Every sample requires all four neighbours to lie on
observed remapping support. The right patch is centred at left x minus the fixed
predicted disparity. No displacement, camera correction or patch warp is optimized.
Each patch assumes constant disparity, which can be wrong across depth boundaries.

Signed zero-mean normalized cross correlation (NCC) removes a positive affine
brightness change; constant patches have undefined NCC and remain unavailable.
All-reference counts, support for each estimator and the common finite subset
are reported separately. Fixed left-patch standard-deviation bins below 2, 2–8
and at least 8 display DN describe texture without selecting by measured error.
Those exploratory bins are not qualification criteria or comparable physical
radiance levels between sensors. No failed reference point is deleted.

For the time-aware cases, these are medians of common-subset frame-median NCC
values across 14 available frames. Higher NCC is stronger local image agreement,
not proven correspondence accuracy. All 16 frames remain in reference counts.

| Sensor | Common patch points | LiDAR predicted disparity | StereoSGBM disparity | RAFT disparity |
|---|---:|---:|---:|---:|
| RGB | 49,373 | 0.82449 | 0.90780 | 0.90119 |
| Thermal | 52,048 | 0.86901 | 0.92450 | 0.91318 |

Paired, pointwise NCC differences are also computed rather than subtracting
these aggregate medians. Both estimators have positive median differences from
LiDAR on all 14 time-aware frames in each sensor. StereoSGBM has the stronger
paired median than RAFT on all 14 frames; median-of-frame paired RAFT-minus-SGBM
differences are -0.00103 for RGB and -0.00124 for thermal. Static cases have the
same broad ranking. In the higher-texture thermal group, NCC medians are 0.93327,
0.97089 and 0.96330 for LiDAR, SGBM and RAFT, respectively.

The adverse control is important: the unrelated-right thermal comparison retains
only seven common patch points in one frame, yet their NCC medians are 0.87417
for SGBM and 0.85277 for RAFT, versus 0.17254 for the LiDAR prediction. No RGB
wrong-right points have common finite support. A high selected-patch correlation
can therefore accompany a false match. Local texture, repeated structures,
occlusion and the estimators' image-matching objectives limit this diagnostic;
it is not independent validation of those objectives.

**Inference and next action:** the LiDAR residual cannot be interpreted directly
as actual image-registration error, and replacing SGBM by RAFT has not established
a better fine-correspondence reference. No unique calibration error is identified.
Before fitting a correction, examine horizontal/vertical patch-score sensitivity
and ambiguity using the retained cases, including the unrelated-image controls.
This can distinguish a coherent geometry discrepancy from ambiguous patch matches;
it cannot justify declaring the best local correlation to be GT. No GPU training,
reference filtering or generator authorization follows from this result.

The implementation is `scripts/probe_ms2_stereo_patch_evidence.py` with numerical
helpers in `src/aero_ir/registration/stereo_patch_evidence.py`. Report:
`experiments/ms2_stereo_patch_evidence_01/report.json`, SHA256
`74de1eaafdfcbf556b0538d131b381b8880808a742fb0e07f1e67f5f8227da56`.
The checker verified 524 recorded input/source identities, all 96 artifact hashes,
unchanged reference coordinates/depths, all saved summaries and aggregates.
An additional 8,448 deterministic case/point/method samples were checked against
independent SciPy bilinear patch interpolation and NCC reconstruction. This
validates the calculation, not the physical correspondence. Full CPU tests:
**920 passed**, five existing warnings; Ruff code checks and formatting of the
three new files passed. The pre-existing full-repository formatting blocker
remains; frozen inference sources were not reformatted.

To reproduce the CPU diagnostic into a new directory:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/aero-ms2-mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_stereo_patch_evidence \
  --out-dir experiments/ms2_stereo_patch_evidence_repeat01
```

## Horizontal and vertical score profiles

A second CPU diagnostic completed all 96 cases on 2026-10-01. Around each
unchanged LiDAR-predicted right-image position, it evaluates signed patch NCC
at offsets from -3 to +3 rectified pixels in 0.5-pixel increments. Horizontal
and vertical searches are separate, not a joint two-dimensional fit. Positive
x moves the right patch rightward and decreases disparity; positive y moves it
downward. No correction is applied to any image, calibration or disparity map.

The denominator retains every original reference point. A compared profile
requires finite NCC at every offset on both axes, so one shift cannot win merely
by gaining easier border support. Tied maxima within 1e-9 NCC have no reported
unique shift. Endpoint maxima are counted as censored, not extrapolated. A local
peak gap uses the strongest competitor at least 1 pixel away along the same axis;
it is not a probability or a guarantee of globally unique correspondence.

| Time-aware input | Full references | Complete profiles | Median horizontal peak offset | Median vertical peak offset |
|---|---:|---:|---:|---:|
| RGB paired | 98,628 | 81,045 | -0.5 px | +0.5 px |
| Thermal paired | 132,302 | 105,975 | -1.0 px | +0.5 px |
| RGB unrelated right | 98,628 | 81,026 | 0.0 px | 0.0 px |
| Thermal unrelated right | 132,302 | 105,930 | 0.0 px | 0.0 px |

Offset columns are medians of frame medians among unique discrete peaks, over
14 available frames. Each sensor's units are its own rectified pixels, not the
equivalent thermal-disparity scale used in the LiDAR residual comparison.
The two unavailable endpoint poses stay in reference counts. All 14 paired RGB
frames have median vertical peak +0.5 px. Thermal has +0.5 px in 10 frames and
+1.0 px in four. Both sensors retain a +0.5 px aggregate vertical peak in the
predeclared higher-texture group. The horizontal thermal frame medians vary from
-2.0 to 0.0 px, which does not support simply applying one global horizontal shift.

The distributions are not all concentrated at their median. Paired thermal
profiles have 30,390/105,975 horizontal endpoint maxima and 9,784/105,975 vertical
endpoint maxima. Unrelated thermal pairs have 64,062/105,930 and 54,108/105,930,
respectively. Their median separated vertical peak gap is 0.05263 NCC, larger
than 0.03957 for real pairs. Wrong pairs can therefore have a seemingly distinct
local optimum; neither the optimum nor its local gap is an automatic GT test.

**Interpretation:** a repeated vertical preference is a measurable candidate
effect, not a diagnosed camera-calibration error. Horizontal anchoring errors,
slanted or repeated texture, and interpolation can couple into a vertical score
profile. The following controlled check repeats the vertical comparison with
SGBM and RAFT horizontal anchors instead of LiDAR. A shared +0.5-pixel preference
in different sensors must not be
equated with the same angular miscalibration. Anti-UAV qualification and dense
generator eligibility remain unchanged.

Report: `experiments/ms2_stereo_patch_profile_01/report.json`, SHA256
`f278c66d0d0b2c36635f95bfdd4be3c11d6bef673d44770bf0afce2dfd390fe0`.
Verification checked 624 input/source identities, all 96 artifact hashes and
reference arrays, exact agreement with the preceding experiment at zero offset,
reconstructed peak/tie/endpoint counts, all summaries and aggregates. Known-shift
tests establish horizontal sign, vertical direction, tie handling, separated
competitors and missing-profile preservation. Full CPU suite: **927 passed**,
five existing warnings; Ruff code checks and formatting of the three new files
passed. No existing frozen source or manifest was modified.

CPU reproduction into a fresh directory:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/aero-ms2-mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_stereo_patch_profile \
  --out-dir experiments/ms2_stereo_patch_profile_repeat01
```

## Vertical profiles with alternative horizontal anchors

The completed 96-case CPU follow-up finds that the +0.5-pixel vertical preference
persists when SGBM or RAFT supplies the horizontal position. It is therefore not
specific to the LiDAR disparity anchor. This is evidence for testing a small
vertical image-compensation candidate, not proof of an erroneous calibration
parameter or registration qualification.

The comparison fixes each method's saved disparity and evaluates the unchanged
-3 to +3 pixel vertical grid. Every original reference remains counted, while
reported peak comparisons use the exact same points with complete profiles for
all three anchors. Zero-offset scores reproduce the preceding patch experiment
for every method and case. No model inference, camera fit or correction occurs.

| Time-aware paired sensor | Full reference points | Common profile points | LiDAR anchor median | SGBM anchor median | RAFT anchor median |
|---|---:|---:|---:|---:|---:|
| RGB | 98,628 | 48,733 | +0.5 px | +0.5 px | +0.5 px |
| Thermal | 132,302 | 51,458 | +0.5 px | +0.5 px | +0.5 px |

These are medians of common-point frame-median vertical peaks across 14 available
frames, in each sensor's rectified pixels. All 14 RGB frame medians are +0.5 px
for every anchor. Thermal SGBM medians are +0.5 px in 12 frames, 0 px in one and
+1 px in one; its RAFT medians are +0.5 px in 11 frames and +1 px in three.
Static cases and the predeclared higher-texture group also retain aggregate
+0.5-pixel peaks for all anchors. Individual points are not all shifted identically:
thermal/time-aware endpoint-peak counts are 2,612 for LiDAR, 1,180 for SGBM and
983 for RAFT within the common 51,458 points.

An execution control uses each actual rectified left image's texture to construct
an exact six-pixel horizontal shift, with known vertical shift either 0 or +1 px.
It tests 64 evenly spaced original reference indices in every case with available
geometry. All 176 case/shift controls pass: the expected vertical location is a
maximum wherever the complete profile is measurable (10,351 of 11,264 sampled
point/control instances). Missing patches remain recorded; these repeated cases
are not 176 independent image pairs. This argues against an unconditional +0.5 px
bias in the patch scoring code. Because synthesis occurs after rectification, it
does not validate the original cameras' calibration or differential preprocessing.

Unrelated-right cases still have no RGB common profiles and just seven thermal
common points in one frame, with inconsistent anchor medians (0, -0.5 and +1.5 px).
They remain insufficient to certify the remaining real-pair matches. Scene content,
interpolation and geometry can still affect the measured preference.

**Controlled candidate evaluated below:** test a fixed +0.5 px right-image sampling offset
before stereo estimation, preserving the source images and calibration. Include
the opposite -0.5 px offset as an equal-interpolation directional control and an
unchanged baseline that must reproduce the saved arrays. Compare full-reference
support and common-point LiDAR discrepancies, not only optimized patch NCC.
This is an exploratory image-space intervention, not an installed physical camera
correction; qualification and generator authorization remain on hold.

Report: `experiments/ms2_stereo_anchor_profile_01/report.json`, SHA256
`a2c9232c965b82f333ea19795b4e8886d72de0208480a8c4738ab143b5f60fc3`.
Verification checked 724 input identities, 96 artifact hashes and unchanged
reference coordinates, all zero-offset scores, common-support summaries and
aggregates, and all 176 saved known-shift control results. Full CPU tests:
**933 passed**, five existing warnings. Ruff code checks and formatting of the
three new files pass; the historical repository-formatting blocker remains.

CPU reproduction into a fresh directory:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/aero-ms2-mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_stereo_anchor_profile \
  --out-dir experiments/ms2_stereo_anchor_profile_repeat01
```

## Vertical sampling intervention

A fixed +0.5 px right-image sampling offset improves the thermal stereo comparison
on this observed training panel, but not RGB common-point accuracy. This is a
candidate for confirmation on different frames, not an installed calibration or
registration qualification. Positive sampling reads the right image half a pixel
downward, moving its displayed content upward. Both half-pixel directions use the
same linear interpolation; zero bypasses resampling. Camera matrices and Q remain
unchanged. Every contributing source row must be observed.

The 96 original cases were each evaluated at 0, +0.5 and -0.5 px. Zero reproduces
all saved baseline stereo arrays and scores. All original LiDAR reference points,
unavailable poses and unrelated-right controls remain included. The common-point
comparison below uses the exact intersection of valid estimates across all three
settings: 44,487 RGB points and 44,926 thermal points across 14 available frames.

| Time-aware sensor | Sampling offset | Supported points | Full-reference fraction within 3 px | Common-point median discrepancy |
|---|---:|---:|---:|---:|
| RGB | 0 px | 52,801 / 98,628 | 50.02% | 0.6694 px |
| RGB | +0.5 px | 54,220 / 98,628 | 51.44% | 0.6831 px |
| RGB | -0.5 px | 48,272 / 98,628 | 45.17% | 0.6649 px |
| Thermal | 0 px | 55,005 / 132,302 | 35.98% | 1.1332 px |
| Thermal | +0.5 px | 56,463 / 132,302 | 37.96% | 1.0190 px |
| Thermal | -0.5 px | 50,349 / 132,302 | 31.51% | 1.2301 px |

Fractions are equal-reference-frame means; discrepancies are medians of frame
medians in equivalent thermal-focal pixels, not raw RGB pixels or pooled-point
medians. Thermal +0.5 improves common-point frame medians in 14/14 time-aware
frames and 14/16 static frames. Static thermal aggregate discrepancy decreases
from 0.9383 to 0.8692 px on 47,517 common points. Opposite-direction time-aware
thermal sampling improves only 1/14 frames. RGB +0.5 improves only 5/14 time-aware
frames, so the result does not justify applying one correction to both sensors.

Unrelated-right estimates remain poor: time-aware common-point median discrepancies
are about 21 px for RGB and 48 px for thermal. Their full-reference within-3-px
fractions remain below 0.11% for every setting. These controls do not certify
individual real-pair correspondences.

The approximately 10% reduction in the thermal common-point aggregate is
exploratory: the offset was selected after inspecting this same panel. It does
not establish a physical calibration cause, dense RGB–IR ground truth, or
generator-training eligibility. The fixed candidate was subsequently compared
with zero and the opposite direction on different training frames below.

Report: `experiments/ms2_vertical_compensation_01/report.json`, SHA256
`e480d336f6320f9968c44eec2e2aa24aede101e6a0a4ee920236a93ca14c8afb`.
Verification checked 727 input identities, all 288 artifact hashes and reference
coordinates, reconstructed depth/disparity scores, all common-point summaries and
aggregates. Full CPU tests: **937 passed**; Ruff code checks passed. This does not
resolve the historical repository-wide formatting blocker.

CPU reproduction into a fresh directory:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/aero-ms2-mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_vertical_compensation \
  --out-dir experiments/ms2_vertical_compensation_repeat01
```

## Disjoint frame confirmation

The thermal-specific benefit reproduces on the fixed 15-frame quarter panel,
without retuning the offset, display window, camera parameters or stereo settings.
These frames do not overlap the offset-selection panel, but were previously used
for other diagnostics and come from the same training sequence. This is not a
blind, independent-scene or held-out-test evaluation.

All 30 original single-scan depth maps were extracted with full archive integrity
verification. The experiment retains every nonzero LiDAR reference and all three
conditions: static pairing, time-aware pairing and a cyclic-seven unrelated-right
control. There are 270 sensor/frame/condition/offset cases. Zero-offset estimates
exactly reproduce the unchanged estimator on these frames. Source images and
camera matrices are not modified.

| Time-aware sensor | Sampling offset | Supported points | Full-reference fraction within 3 px | Common-point median discrepancy |
|---|---:|---:|---:|---:|
| RGB | 0 px | 62,362 / 97,421 | 60.39% | 0.6460 px |
| RGB | +0.5 px | 64,008 / 97,421 | 62.08% | 0.6628 px |
| RGB | -0.5 px | 57,353 / 97,421 | 55.09% | 0.6542 px |
| Thermal | 0 px | 66,825 / 130,625 | 43.99% | 1.0599 px |
| Thermal | +0.5 px | 67,478 / 130,625 | 45.09% | 0.8708 px |
| Thermal | -0.5 px | 62,741 / 130,625 | 39.77% | 1.1657 px |

The common-point intersection across all three settings contains 53,457 RGB and
56,322 thermal points across all 15 frames. As above, discrepancy is the median
of frame medians in equivalent thermal-focal pixels; fractions are equal-frame
means, not pooled point ratios. The thermal aggregate falls by approximately 17.8%,
but individual frame medians improve in only 12/15 frames, with three worsening.
The opposite direction improves only 2/15. Static thermal comparison also improves
(0.9217 to 0.7854 px on 50,749 common points, 14/15 frame medians improving).
RGB +0.5 improves only 5/15 time-aware frame medians and worsens the aggregate.

Unrelated-right controls retain large common-point discrepancies: approximately
17.6 px for RGB and 55.1 px for thermal. Full-reference within-3-px fractions stay
below 0.13% for every setting. Missing estimates are retained as failures in the
full-reference fractions rather than removed from the denominator.

**Decision:** retain +0.5 px as a thermal-only stereo sampling candidate, not a
universal camera correction. Its effect on actual calibrated RGB-to-IR projection
must be checked next, using unchanged RGB depth and camera transforms. Improving
thermal left/right depth alone neither establishes dense cross-modal pixel GT nor
authorizes generator training. Anti-UAV qualification is unchanged.

Report: `experiments/ms2_vertical_confirmation_01/report.json`, SHA256
`c8854f3945f77ad91be273af3fb1e88180af765c7dfddc6980d5eab24a4d89ef`.
Verified summary: `experiments/ms2_vertical_confirmation_01/verified_summary.json`,
SHA256 `70f3782a7851e14b4bbac913588a6624e264c665a5a632d13555c81458cba038`.
The verifier checked 882 input identities, 270 artifact hashes, original depth
coordinates and values, frame pairing, unchanged geometry across offsets,
reconstructed depth/disparity scores and aggregates. This was saved-map rescoring,
not a separate stereo inference replay. CPU tests: **953 passed**, five existing
warnings; Ruff code checks and formatting of the five new confirmation files pass.
The historical repository-wide formatting blocker is unchanged.

The extraction's exact preparation source was preserved in
`experiments/ms2_vertical_confirmation_preparation_frozen.py` before a line-wrap
cleanup. The runner checks the recorded hash against the current source or that
exact historical snapshot; it never substitutes an unverified version.

After the verified depth extraction, CPU reproduction uses fresh output paths:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/aero-ms2-mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_vertical_confirmation \
  --out-dir experiments/ms2_vertical_confirmation_repeat01

CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/aero-ms2-mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.analyze_ms2_vertical_confirmation \
  --run experiments/ms2_vertical_confirmation_repeat01 \
  --out experiments/ms2_vertical_confirmation_repeat01/verified_summary.json
```

## Fixed projection cross camera check

The thermal sampling candidate produces only a modest improvement when evaluated
at the original calibrated RGB-to-IR projected points. It does not change the
RGB-to-IR mapping itself. The 17.8% same-camera thermal aggregate improvement
must therefore not be reported as a 17.8% cross-modal registration improvement.

This experiment reuses all 45 author-camera frame/condition cases from the frozen
cross-stereo comparison. RGB source depth, projected thermal coordinates, expected
thermal Z-depth, camera geometry and reference masks remain exactly unchanged.
Only the rectified right thermal image receives 0, +0.5 or -0.5 px sampling before
stereo estimation. All zero-offset arrays and depth scores reproduce the original
baseline; there are 135 resulting cases, with no camera fitting or new GPU inference.

| Time-aware thermal sampling | Supported points | Full-reference fraction within 3 px | Common-point median disparity discrepancy | Improved frame medians |
|---|---:|---:|---:|---:|
| 0 px | 11,281 / 16,583 | 59.98% | 1.2055 px | Reference |
| +0.5 px | 11,291 / 16,583 | 60.87% | 1.1621 px | 8 / 15 |
| -0.5 px | 10,618 / 16,583 | 55.27% | 1.2540 px | 4 / 15 |

All three settings share 9,999 measurable points in the time-aware comparison.
Fractions are equal-frame means and discrepancies are medians of frame medians
at the author thermal focal scale. The approximately 3.6% aggregate reduction is
not a cross-modal pixel localization error reduction: depth can agree at wrongly
associated points on equal-depth surfaces. A new execution test explicitly
demonstrates that ambiguity rather than treating depth agreement as a qualifier.

Static cases share 8,997 points; +0.5 reduces their aggregate from 1.1982 to
1.0994 equivalent pixels and improves 12/15 frames. Unrelated-right controls
retain large common-point discrepancies (about 28 px) and full-reference within-3-px
fractions below 0.15%. These controls and improvements do not establish correct
individual correspondences.

**Decision:** retain the measured thermal stereo improvement as a diagnostic
finding, but do not promote it to a registration fix or launch more same-camera
training as the solution. The missing evidence remains spatial RGB–IR correspondence.
No generator export, scientific qualification or Anti-UAV approval changes.

Report: `experiments/ms2_cross_vertical_sampling_01/report.json`, SHA256
`d4f776e0b7ad16c62dd394bbbd94e40dfdf272404cae3e3b9c42e497f4bf9238`.
Verification checked 1,218 source identities, all 135 artifact hashes, unchanged
source depth/projection/reference arrays and geometry, reconstructed scores,
common-point summaries and aggregates. Full CPU tests: **957 passed**, five
existing warnings. Ruff code checks pass; the historical formatting blocker remains.

CPU reproduction into a fresh directory:

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/aero-ms2-mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_ms2_cross_vertical_sampling \
  --out-dir experiments/ms2_cross_vertical_sampling_repeat01
```
