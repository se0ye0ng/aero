# MS² RGB-stereo depth and fixed-calibration confirmation

Status:2026-09-24. CPU depth-source comparison, exploratory calibration fits and
the frozen15-frame confirmation are complete. Both rotation-only candidates
improve all15new frames for both matchers. **Registration and generator training
are not approved.** This is a
separate additional-source investigation, not a repair/pass on Anti-UAV300.

## Question and independent input

The [initial image screen](registration_ms2_image_screen.md) found several-pixel
disagreement between two frozen image matchers and calibrated LiDAR-depth
projections. This diagnostic changes the depth source: estimate depth from **only
the left and right RGB images**. Neither IR images, RGB–IR matches, LiDAR depth,
nor cross-modal reference errors enter stereo disparity estimation.

OpenCV StereoSGBM is a block-based stereo algorithm with disparity smoothness and
matching checks; it does not return measured ground truth. Its disparity settings
and pixel units follow the [OpenCV documentation](https://docs.opencv.org/4.13.0/d2/d85/classcv_1_1StereoSGBM.html).
The actual installed runtime is OpenCV5.0.0; synthetic-disparity tests separately
verify the used sign and fixed-point conversion in that runtime.

Settings are frozen before real-data computation: native1224×384 grayscale RGB,
128disparities from0, block size5, P1=200, P2=800, uniqueness10, speckle window100,
speckle range2, prefilter31 and SGBM mode. Disparity is divided by16. Both views
are mirrored/swapped for an explicit reverse estimate. Positive disparities must
agree within1pixel across views. At each queried source position all four
bilinear neighbors must be supported and have disparity spread≤1pixel. Missing
depth is never filled. These are fixed diagnostic support rules, not qualification
thresholds chosen from the cross-modal result.

Depth is `focal * baseline / disparity`, using supplied rectified RGB intrinsics
and the0.299184m stereo baseline. The runner checks equal left/right intrinsics,
identity relative rotation and horizontal translation. Selected RGB stereo
timestamp skew has maximum absolute magnitude0.081316ms. Comparing static and
time-compensated right-camera projections at the source LiDAR anchors gives a
maximum0.07685pixel displacement. The diagnostic reports this residual rather
than claiming exactly simultaneous exposures.

## Actual depth-source comparison

The first image with a known synthetic12pixel disparity yields375,936supported
pixels out of470,016; all supported disparities are within1pixel of the known
value, with median error0. Border/unmatched pixels remain unsupported. This is
an execution control, not RGB–IR evidence.

On the16real frames, dense RGB-stereo support ranges from51.46% to72.18%, with
median61.76%. At the original nonzero LiDAR pixels, the median-of-frame median
stereo-versus-LiDAR disparity discrepancy is1.418pixels. This establishes a
disagreement, not which source is correct near edges, occlusions or moving objects.

The table compares the **same RGB–IR match points** for which both depth sources
produce supported thermal projections. PCK3 means distance to the matched thermal
point≤3native thermal pixels. It is an equal-frame average, with all16frames
represented; median error is the median of frame medians. These are raw match
comparisons, not the original sparse-reference TPS score.

| Matcher | Common supported points | LiDAR-depth median error | RGB-stereo-depth median error | LiDAR PCK3 | Stereo PCK3 |
|---|---:|---:|---:|---:|---:|
| XoFTR640 | 139 | 4.557px | 4.180px | 24.40% | 26.10% |
| MINIMA-XoFTR | 111 | 4.045px | 3.820px | 31.79% | 40.02% |

Across **all stereo-supported matched points**, not just the tiny intersection,
XoFTR has13,373/18,793supported points and median4.343px; MINIMA has11,281/15,233
and median3.797px. Their equal-frame, support-conditional PCK3 is24.07%/29.01%.
Full-match denominators and unsupported counts are also retained in the JSON;
these conditional percentages must not be presented as dense coverage.
Unrelated-thermal controls have conditional median errors70.257/59.224px and
conditional PCK3 of0.644%/0%, respectively. Thus meaningful image-pair structure
is present, but replacing LiDAR depth alone does not solve registration.

Report: `experiments/ms2_stereo_depth_01/report.json`, SHA256
`1b743c90200668914df30efbe90610af803be3340d21e1135027e45be101d5a4`.
Runner: `scripts/probe_ms2_stereo_depth.py`. All source images, native projected
depths, calibration and earlier reports remain unchanged.

## Bounded fixed-rig fit with the new depth source

Repeat the prior exploratory fitting protocol unchanged except for source depth:
even-indexed8frames fit, odd-indexed8frames check; rotation-only and rigid6
candidates; ±3degrees per rotation-vector component and±0.1m per translation
component; soft-L1 loss at1pixel scale. Image-stereo support is independent of
cross-modal error magnitude. Both matcher-specific fits are evaluated on both
matchers' saved correspondences. The check frames were previously inspected, so
this is **not unseen official-test performance**.

| Matcher, own correction,8check frames | Supported points | Author PCK3 | Rotation-only PCK3 | Rigid6 PCK3 | Author / rotation / rigid median error |
|---|---:|---:|---:|---:|---:|
| XoFTR640 | 6,663 | 33.96% | 54.21% | 54.62% | 3.814 /2.626 /2.793px |
| MINIMA-XoFTR | 5,551 | 40.82% | 64.02% | 62.46% | 3.295 /2.225 /2.516px |

The rotation-vector estimates are similar across the two independently fitted
checkpoint outputs: XoFTR[-0.5306,-0.1611,0.2233]degrees and
MINIMA[-0.5007,-0.1949,0.2156]degrees. The checkpoints share an architecture;
their agreement must not be called fully independent correspondence ground truth.
Both rigid6fits reach the+0.1m target-Z search boundary, so neither is selected
as a validated physical correction. The two rotation-only candidates are frozen
for a new-image confirmation, **not installed into the dataset or training pipeline**.

Report: `experiments/ms2_stereo_calibration_01/report.json`, SHA256
`c70deb86993fe8c3f1140da3d7f3a35233ea3593cb10b9a3a0b821d61d1cacc4`.
Runner: `scripts/probe_ms2_stereo_calibration.py`. Bounds, optimizer and8/8split
are recorded before optimization. No fit changes qualification authorization.

## Frozen confirmation protocol and completed extraction

Plan: `experiments/ms2_confirmation_plan_01/plan.json`, SHA256
`32c1c923654a46ddfe807905dc4981587d1c1861d8f2a07e81ecdaea9a4f4f1d`.
Choose the integer midpoint of every consecutive pair in the original16-frame
list. This yields15previously uninspected image IDs, with no overlap:

`000348 001044 001740 002436 003132 003828 004524 005220 005916 006612 007308 008004 008700 009396 010092`

The same official **training sequence** is used. This is a test of transfer to
new frame IDs, not evidence of generalization to another sequence, camera rig,
official validation split or official test split. Candidate matrices are copied
into the plan **before** extracting these frames. No refitting, frame exclusion,
thermal-window adaptation or post-result threshold selection is allowed.

`scripts/prepare_ms2_confirmation.py` validates the original archive identity,
then reuses the safe bounded extractor and full bzip2CRC/EOF verification. The
completed extraction used already downloaded data, not another network transfer.
Do not start scoring partial extracted files or treat a present directory as a
completed extraction. The complete report is required by the runner.

The extraction completed with117verified files and no missing requirements.
Report: `experiments/ms2_confirmation_sync_01_report.json`, SHA256
`400a4732f443c1c9d2b2785519e83bc1b40d7e6d71ea2041aaee384d5e933b96`.
The extraction source hashes and every selected file were rechecked before
launching fixed-candidate inference. No partial extraction was used for scoring.

The confirmation runner evaluates the author calibration and both frozen
rotation-only candidates with both fixed matchers and an unrelated-thermal
control(cyclic offset7). It keeps the original3308–4974DN thermal window and
same RGB-stereo settings. Candidate comparisons share the fixed original-
calibration support mask, while all-match fractions retain unsupported pixels.
Improvement in this confirmation does not by itself certify physical pixel GT,
bidirectional dense support or a generator export. It justifies a further
independent validation stage rather than another unconstrained300epoch run.

CPU commands, using fresh output destinations. The confirmation `repeat01`
run is already complete; the examples below use `repeat02` to avoid overwriting
its evidence. Repeating these commands is optional, not an unfinished step:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.probe_ms2_stereo_depth --out-dir experiments/ms2_stereo_depth_repeat01
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.probe_ms2_stereo_calibration --out-dir experiments/ms2_stereo_calibration_repeat01
# Only after experiments/ms2_confirmation_sync_01_report.json is complete:
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.probe_ms2_confirmation --out-dir experiments/ms2_confirmation_image_repeat02
# Only after the inference report is complete:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.analyze_ms2_confirmation \
  --report experiments/ms2_confirmation_image_repeat02/report.json \
  --out experiments/ms2_confirmation_analysis_repeat02/report.json
```

No GPU training, Git commit or GitHub push is part of these diagnostic runs.

Verification:829CPU tests passed with five existing dependency
warnings; repository Ruff and `git diff --check` passed. All286depth-diagnostic
dependencies,321calibration-diagnostic dependencies, saved match/disparity arrays,
and both preflight hashes were rechecked. Confirmation extraction and inference
also completed; their verified results are below.

Before confirmation inference, the score implementation was factored into a
tested fixed-denominator function. Projections lost after correction remain
failures; they do not shrink the comparison population. The new confirmation
analyzer verifies the exact planned case set and every saved artifact, recomputes
depth samples and candidate reprojection scores, and reports per-frame gains,
regressions, failures and support. Empty frames remain explicitly represented;
the full-panel summary scores them as zero and labels that convention. This is
saved-data score reconstruction, not model replay or physical-GT verification.
Confirmation-input identity insertion now rejects changes to already frozen
dependencies instead of replacing their earlier hashes. The extraction archive
identity is checked against the acquired source explicitly. These checks protect
experiment identity and arithmetic; they do not authorize generator training.

## Confirmation results on15new frames

The completed run has60case rows:15frames×2matchers×(paired/unrelated thermal).
Neither calibration candidate was refitted. All15paired frames have stereo
reference support. Both fixed corrections improve PCK3 on **15/15frames for each
matcher**, not merely the pooled average. No candidate loses projected support
on the fixed comparison population.

| Evaluation matcher | Fixed correction | Stereo-supported / all image matches | Equal-frame PCK3 | Median of frame median errors | Improved / worsened frames |
|---|---|---:|---:|---:|---:|
| XoFTR640 | Author calibration | 12,511 /17,287 | 21.59% | 4.351px | — |
| XoFTR640 | XoFTR-fitted rotation | 12,511 /17,287 | 54.39% | 2.544px | 15 /0 |
| XoFTR640 | MINIMA-fitted rotation | 12,511 /17,287 | 54.46% | 2.396px | 15 /0 |
| MINIMA-XoFTR | Author calibration | 10,703 /14,229 | 30.42% | 3.751px | — |
| MINIMA-XoFTR | XoFTR-fitted rotation | 10,703 /14,229 | 67.08% | 2.000px | 15 /0 |
| MINIMA-XoFTR | MINIMA-fitted rotation | 10,703 /14,229 | 67.22% | 2.026px | 15 /0 |

PCK3 uses3native thermal pixels and the **same original-geometry-supported points**
for every candidate. Unsupported raw image matches are not included in this
conditional table; all-match scores are also retained in the full report. No
frame is excluded. The two fixed corrections give similar scores even when
evaluated on the other matcher's points. That supports a repeatable geometric
correction, but the shared matcher architecture and estimated stereo depth mean
this is not absolute physical-pixel ground truth.

Unrelated-thermal controls remain poor. XoFTR fixed-support PCK3 changes from0.464%
to0.232% with either correction; MINIMA remains0%. Thus the gain is not an
indiscriminate improvement of incorrectly paired images. Remaining variation is
substantial: own-correction per-frame PCK3 ranges27.32–68.21% for XoFTR and
37.61–81.83% for MINIMA. In particular, improving all frames is **not** the same
as passing the original95%Anti-UAV frame-qualification criterion; the datasets,
denominators and metrics are different.

The original thermal window was preserved. Clipping on the new frames ranges
0.00061–3.489% of thermal pixels; no frame-specific normalization was selected.
RGB-stereo support covers51.49–71.25% of native RGB pixels(median62.89%), so this
does not establish dense observed support for an unmasked generator loss.
The new panel's maximum absolute RGB left/right timestamp skew is0.171234ms.

Artifacts:

- Inference: `experiments/ms2_confirmation_image_01/report.json`, SHA256
  `08ef4d8f40d05a02671ac92bbfd87e948b5cd11852dd54a574f8f470327a0843`.
- Pre-inference identities/protocol: `preflight.json` in that directory, SHA256
  `15177a615cce371bde30fc3fbbf671b8c88e1accea9acaa51b2a31cbf773aaa0`.
- Recomputed analysis: `experiments/ms2_confirmation_analysis_01/report.json`,
  SHA256 `7061117818ebd5250a59ca6c60b9087effe9f8619b94ef659cabe1c3795726bd`.

The analyzer checked the planned case set and518dependency/artifact identities,
resampled stored stereo disparities, reconstructed candidate projections and
recomputed every saved score. It did not rerun neural inference or independently
measure physical pixel locations. The next problem is the remaining fine-scale
error and unsupported observation domain, not a missing completed training run.
No calibration is installed as qualified, no generator is trained, and no
Anti-UAV failure is relabelled as a pass.

## Completed independent-process inference replay

A separate CPU process reran both frozen matchers from the images for all60
planned cases, and recomputed the15 RGB-stereo disparity artifacts. This is
distinct from the saved-data score reconstruction described above.

The original and replay output directories contain77 matching files:60 match
artifacts,15 stereo artifacts, the preflight and the inference report. All77
are byte-identical by SHA256 comparison. The inference report at
`experiments/ms2_confirmation_image_repeat01/report.json` therefore retains SHA256
`08ef4d8f40d05a02671ac92bbfd87e948b5cd11852dd54a574f8f470327a0843`.
The repeated analysis has identical aggregate results; its report at
`experiments/ms2_confirmation_analysis_repeat01/report.json` has SHA256
`c866c94fc8a6d7fd6432d53044f5e22d2493a8ae93e885b27f86d409a19e80b5`.
Its file identity differs because the recorded verified artifact paths point to
the replay directory.

This establishes repeatability for this CPU environment, fixed inputs and
protocol. It does not establish physical accuracy, cross-sequence transfer or
dense registration qualification. Further changes fitted using these now-observed
frames require a new frozen confirmation panel; they cannot be presented as
unseen confirmation on this same panel.

The subsequent [residual attribution and intrinsic sensitivity
study](registration_ms2_residuals.md) tests why fine-scale errors remain. It is
explicitly exploratory reuse of the observed panel, not a replacement for this
frozen confirmation or a registration approval.
