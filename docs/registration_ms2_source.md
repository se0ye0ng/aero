# Additional paired training source: MS² acquisition and preflight

Status, 2026-09-24: **candidate, not qualified**. The user authorized validation of
an additional public RGB–IR generator training source, then supplied the authors'
three download-script links after requesting access. Anti-UAV300 repair remains
in scope. Success on MS² must never be reported as Anti-UAV success.

## Why this candidate

The authors provide RGB/thermal images, calibration, timestamps, LiDAR-derived
depth and odometry. These enable geometry checks not based solely on the matcher
being evaluated. [Official dataset](https://sites.google.com/view/multi-spectral-stereo-dataset)
and [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Shin_Deep_Depth_Estimation_From_Thermal_Image_CVPR_2023_paper.html).
This is a road-driving domain, not a UAV tracking dataset.

The website specifies CC BY-NC-SA 3.0 for its datasets and benchmarks. Keep the
data separate from the project's MIT code; preserve attribution and review the
applicable terms before publishing derived assets. No raw data, access-script
contents or model weights are being added to Git. The public metadata loader is
read for format interpretation, **not imported or executed**.

Other candidates checked in this acquisition pass:

- LLVIP: official terms and download instructions saved at repository commit
  `c1a655cce437ebfd990a97b04fc48fbb99f4c47b`. The official Google Drive download
  attempt returned a download-quota error; no image archive was obtained. Its
  author-provided projective alignment is not independently measured residual
  accuracy. No unverified mirror was substituted.
- [Breaking Modality Disparity](https://arxiv.org/html/2304.05646v2), §4:
  distinguishes synthetic perturbation references from 105 real pairs with manual
  calibration. A synthetic deformation field cannot certify the underlying
  cross-modal physical alignment. No usable author download/usage contract was
  established in this pass; no data acquired.
- [Visual-Thermal Camera Dataset](https://www.autonomousrobotslab.com/vtdataset.html):
  the inspected release page says “Under construction”; it did not provide the
  advertised calibrated data through that page. No data acquired.

## Pinned metadata and actual CPU result

`scripts/fetch_ms2_metadata.py` fetches nine fixed Git blobs from
`UkcheolShin/MS2-MultiSpectralStereoDataset`. It verifies the Git blob hash and
records SHA256; downloaded Python is never executed. The observed upstream **tree**
is `0bb0fc1a544d8ad58c7fda5bc025cb0cbb400284` (not a claimed commit hash).

`scripts/audit_ms2_metadata.py` checks every pinned input, duplicate/unsafe
sequence IDs, split intersections and an optional local train-only inventory.
It reads only the header of `calib.npy`, never its pickled object payload.
Metadata success is not registration or generator approval.

| Official split | Sequences |
|---|---:|
| Train | 8 |
| Validation | 3 |
| Test, day | 3 |
| Test, night | 3 |
| Test, rain | 3 |

All ten pairwise split intersections are empty. This establishes ID separation,
not independent routes, places or camera calibration sessions. No held-out pixels
or held-out depth maps were read.

Actual report: `experiments/ms2_metadata_audit_01/report.json`.
SHA256: `7cbb9e6a5c74a825210d0d63c7913ece234f40ac2223509d2dc206657c687242`.
Ten new unit tests passed, including missing-input, duplicate-ID, modified-blob
and object-payload-not-loaded cases. The report does not contain real geometry
measurements. It must not be used as an approved generator manifest.
The full repository test run passed720 tests (five dependency/API warnings);
repository Ruff and `git diff --check` also passed.

## Bounded acquisition

Before opening images or model scores, select the lexicographically first official
train sequence: `_2021-08-06-10-59-33`. User-provided download scripts were fetched
as text and inspected, not executed. They contain `wget` calls and unrestricted
`tar` extraction of all 20 sequences. Only the selected train sequence is fetched
here, under ignored `experiments/external/ms2_first_train_archives/`.

Observed script SHA256:

- `sync_data`: `f923c9280dec4477d7a05d407b98a9eda1933ea2c03811a618f471e7ef2795dc`
- `proj_depth`: `387e078ce81d582be38997917713942ced7dfff1ab289b57fbceb0bcbf6693ca`
- `odom`: `41b35c9b290c8e9d818335d3c81887928347b405455b0c5f3ea18b026314526c`

These are locally observed identities, not separately published checksums. The
archive endpoints support byte ranges. Initial 16 MiB prefixes of the two large
archives were inspected as streams; the small odometry archive was downloaded.
An incomplete `.part` or `.prefix` archive is **not** a completed dataset. Do not
run an unrestricted `extractall` on downloaded archives. Validate member paths,
types, sizes and contents before writing selected data into a fresh destination.

## Coordinate and evidence contract to check on the actual data

The official [format description](https://sites.google.com/view/multi-spectral-stereo-dataset/data-format)
and [setup](https://sites.google.com/view/multi-spectral-stereo-dataset/setup)
must be interpreted separately from model results:

1. Depth PNG values represent metric depth multiplied by256. Zero remains missing
   depth, not a measured surface. Thermal uint16 sensor counts are **not** depth
   and must not be divided by256. Raw counts alone do not guarantee calibrated
   scene temperatures; do not use generic camera-temperature constants as truth.
2. The official loader converts extrinsic translations from mm to m. Its
   composition is `T_thr_to_rgb = T_nir_to_rgb @ inverse(T_nir_to_thr)`.
   Verify the actual per-sequence matrices, conventions, rectification and crop
   offsets before applying them. Equal image indices are not proof of equal
   exposure times. Preserve timestamp integers and measure cross-sensor skew.
3. Use the single-scan measured depth path separately from the multi-frame and
   filtered products. The authors merge32 LiDAR frames and use a learned stereo
   network in filtering; the resulting depth is not an error-free, independent
   dense reference. Motion, occlusions and sparse coverage must be accounted for.
4. Reprojection must use depth and calibrated geometry, not a single homography
   assumed valid at all depths. Check positive target depth, image support,
   occlusion, temporal alignment and covered-pixel fraction. Missing depth must
   not become interpolated “GT.” A sparse successful check does not approve all
   pixels for paired generator loss.
5. Before scoring a registration candidate, freeze a dataset-specific measurement
   protocol with error units, uncertainty, supported regions, exclusions and
   acceptance criteria. Camera self-reprojection/cycle closure is a numerical
   control, not independent accuracy evidence. Validate against available
   additional measurements and account for common calibration errors.

These are preparation requirements, not a claim that the data meets them. No new
human annotation campaign is requested. No Anti-UAV threshold was relaxed, no
failed pair removed, and no GPU training was launched.

## CPU reproduction

```bash
.venv/bin/python -m scripts.fetch_ms2_metadata
.venv/bin/python -m scripts.audit_ms2_metadata \
  --out experiments/ms2_metadata_audit_repeat01/report.json
```

Once a validated extraction exists, add `--data-root /absolute/path/to/MS2dataset`
and use a fresh report path. The local preflight checks only the designated train
sequence; it does not open validation/test images or load calibration pickles.

## First-sequence odometry inspection and frozen frame plan

The small odometry archive completed, with 10,442 records in each of RGB, NIR,
thermal and LiDAR. All four filename ID sets are equal. Every inspected record
contains a finite4×4 matrix and a valid homogeneous final row, although the website
describes3×4 poses. No common world-frame convention or synchronization is inferred
from filename equality. The source bytes remain unchanged.

An exploratory numerical guard measures `max(abs(R.T @ R - I))` and
`abs(det(R)-1)` with tolerance1e-5. This is **not** a physical registration acceptance
threshold. The initial strict check stopped at RGB frame003432. The completed
inspection instead records all violations without dropping or correcting any
frames; this permits acquisition planning, not use of inaccurate poses as GT.

| Modality | Maximum orthogonality deviation | Maximum determinant deviation | Guard violations |
|---|---:|---:|---:|
| RGB | 1.88328e-5 | 1.88326e-5 | 1 |
| NIR | 0.425751 | 0.425809 | 1 |
| Thermal | 8.41091e-6 | 8.40500e-6 | 0 |
| LiDAR | 7.20785e-6 | 7.20598e-6 | 0 |

NIR frame004747 has determinant0.57419055, unlike its two neighboring frames
(approximately1). This is not just the small RGB deviation. Its cause is not yet
established. Do not silently orthogonalize it, interpolate over it or use it as
an independent accurate pose. No affected frame is removed from the source
inventory. This does not establish that RGB/thermal imagery is misregistered;
the NIR odometry stream and the static NIR-to-camera calibration are distinct.

`scripts/prepare_ms2_screen.py plan` freezes16 uniformly spaced integer ranks
over the complete equal ID sets, including the endpoints, independently of pose
guard results or image/model scores. The chosen frame IDs happen not to include
either outlier; selection was not changed to accomplish that. This is a small
training-only development panel, not approval for all frames or a held-out result.

Plan: `experiments/ms2_train_screen_plan_01/plan.json`.
SHA256: `681cf9c6f3d680cf4d0ee7c3e37575868a56066fa3f1bf19662466dedb2952d0`.
Odometry archive SHA256:
`f587847a39952eb03a1dc1164b554d65cfdd8001adfb244bf9976597619ad168`.
The locally recorded archive digest is not an independently published checksum.
The plan explicitly records `pose_numerical_guard_passed=false`, unchanged pose
values, unchecked timestamps and no registration/generator approval.

The accompanying `extract` command processes only a completed archive with the
expected byte count. It verifies bzip2 stream completion, rejects traversal,
links, duplicate members and excessive sizes, writes only selected images,
single-scan depth and supporting metadata to a fresh directory, checks required
outputs, and records input/output hashes. It never executes a downloaded script
or unpickles calibration. Original-resolution pixels are retained without
normalization or visualization conversion. Extraction completion is not geometry
qualification. The two large archives are still acquisition prerequisites.

```bash
.venv/bin/python -m scripts.prepare_ms2_screen plan \
  --archive experiments/external/ms2_first_train_archives/odom.prefix.tar.bz2 \
  --out experiments/ms2_train_screen_plan_repeat01/plan.json
```

Despite its historical `prefix` filename, the odometry file above is the complete
4,189,868-byte download; the other two `prefix` files are incomplete and cannot be
used as complete archives. The new archive-reader tests include truncated streams,
unsafe paths, links, duplicate members, missing frames and immutable output paths.
After adding the archive reader, all736 repository tests passed with CUDA hidden;
repository Ruff and `git diff --check` passed. The successful tests validate the
reader and controls, not the pending real RGB–thermal geometry.

## Calibrated reprojection preparation

`src/aero_ir/registration/calibrated.py` is a separate CPU/NumPy geometry module;
it does not alter the v4 learned-flow convention or any saved registration result.
Its explicit contract is native pixel centres, source-camera axial Z in metres,
and a `T_target_from_source` homogeneous transform. The MS² depth decoder only
applies the documented uint16/256 scale and preserves zero-depth holes as NaN.
Whether the actual projected depth encodes axial Z rather than Euclidean range
must be checked against the supplied projection/calibration evidence before use.
The decoder must never be applied to uint16 thermal intensity images.

The module implements mm-to-m extrinsics, common-camera transform composition,
crop-then-resize intrinsics including the half-pixel offset, point reprojection,
support masks and a projected-point z-buffer. No calibration orthogonalization,
temporal correction, distortion correction, depth densification or image warping
is silently performed. Input intrinsics must refer to the already-rectified
images' actual crop/resolution; a crop adjustment must not be applied twice.

The z-buffer rejects farther samples projected into the same nearest-centre
pixel. An accepted sample is only a **visibility candidate**: an occluder absent
from the sparse point set is not observable through this check. Coverage, time
alignment and independent calibration accuracy remain separate measurements.

Fourteen synthetic unit tests cover transform direction/order, depth-dependent
parallax, round trips with target Z, invalid depth, crop/resize equivalence,
off-image/behind-camera points, z-buffer collisions, empty input and rejection of
improper matrices. A randomized identity-at-boundaries control exposed floating
point residuals up to1.14e-13 pixels; the support predicate therefore uses an
explicit1e-9-pixel numerical epsilon without changing point coordinates. A point
1e-6 pixels outside remains unsupported. This numerical epsilon is not a relaxed
physical registration threshold. None of these tests establishes real MS² or
Anti-UAV registration accuracy or enables generator training.
With this module added, the complete CPU test suite passed750 tests (five existing
dependency/API warnings), including the original v4 coordinate tests. Ruff and
`git diff --check` passed. No real-image reprojection result is claimed yet.

The depth download connection terminated once before completion. Its confirmed
terminal handle was replaced with a byte-range resume, and the HTTP206 response
confirmed continuation at byte5,653,773,481 of19,751,824,021. Subsequently the sync
transfer reached its configured time limit and terminated at12,177,014,046 bytes.
Only after that terminal result was confirmed, it was resumed; HTTP206 confirmed
the next byte and the24,150,818,939-byte total. Neither live transfer was restarted.
Partial downloads remain marked `.part` and are not passed to extraction as
complete inputs.

## Archive metadata preview and upstream caveats

A bounded16MiB suffix request of the sync archive returned bytes
24,134,041,723–24,150,818,938, with the same observed ETag as the main transfer.
Nineteen complete bzip2 blocks were recovered and individually CRC-checked. A
checksum-valid tar header and complete2,715-byte root `readme.txt` were found;
no calibration payload was found in this suffix. This preview is not validation
of the complete archive and must be compared with the full extracted member.
Readme SHA256:
`4fcf504014c7d62a3382b8247771c3a5d7232c457a008503102a4705ad358d16`.
Local partial provenance: `experiments/external/ms2_sync_tail_01/metadata_preview.json`.

The actual readme confirms mm translations and the NIR-left-to-RGB-left and
NIR-left-to-thermal-left transform descriptions. It does not resolve whether
additional rectification conjugation is needed or explicitly distinguish axial
camera Z from Euclidean range. A prefix inspection also found10,442 integer
timestamps each for GPS/IMU and RGB-left; thermal timestamps have not yet been
compared. Timestamp precision must not be lost by parsing Unix nanoseconds as
floating point.

Upstream issue reports inspected on2026-09-24 add reasons to verify actual data:

- [Issue13](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset/issues/13)
  reports unsuccessful NIR-to-thermal reprojection and requests projection code.
- [Issue15](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset/issues/15)
  reports poor cross-modal rectification involving thermal images.
- [Issue7](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset/issues/7)
  questions the paper's FOV values relative to the supplied camera matrices.
- [Issue5](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset/issues/5)
  reports an extraction failure in the same first training sequence selected here.

These are user reports, not confirmed author diagnoses or proof that our files
are defective. No resolution was visible in the inspected page content. Preserve
the frozen sequence and sample selection; check full-stream integrity, actual
image dimensions, calibration conventions and reprojection before claiming a
usable additional training source. Do not replace supplied calibration with
paper FOVs, choose conventions by a desired qualification score, or silently
salvage an archive while labelling it complete.

The depth transfer subsequently exited successfully at exactly19,751,824,021
bytes. Its selective full-stream extraction is running; successful transfer is
not yet a successful bzip2 integrity result. Sync image acquisition remains in
progress. These are only the predetermined first training sequence, not the
complete MS² release.

`src/aero_ir/data/ms2_timestamps.py` adds strict integer timestamp parsing and
same-index skew diagnostics. It subtracts positive int64 timestamps before
converting durations to milliseconds, records duplicate/backward time steps,
rejects count mismatch and never re-pairs images. All13 targeted tests passed,
including1ns differences at Unix-epoch magnitude and extreme-int64 differences;
repository Ruff passed. No RGB-to-thermal skew measurement is claimed yet, and
even equal timestamps would not independently prove exposure synchronization.
The subsequent complete CPU suite also passed763 tests with five existing
dependency/API warnings. Data extraction and real geometry checks remain separate
from these software test results.

## Native panel inspection after extraction

`scripts/audit_ms2_screen.py` requires both completed extraction reports, verifies
their frozen-plan identity and every extracted file hash, then reads the selected
native PNG arrays, calibration and RGB/thermal timestamps. It reports image
dimensions, native dtypes, measured-depth coverage and distributions, numerical
matrix validity and same-index timestamp skew. It does not adjust calibration,
re-pair frames, interpolate depth or approve training. Real-data execution awaits
both full extraction reports; a partial extraction cannot pass this reader.

Calibration dictionaries use NumPy object serialization. The new
`src/aero_ir/data/ms2_calibration.py` reader substitutes inert containers for the
small allowed NumPy pickle constructors and rejects all other globals. It checks
bounded shapes and byte lengths before allocating numeric matrices, rather than
calling unrestricted `np.load(..., allow_pickle=True)`. It supports only the
documented K/R/T matrix dictionary with float32/float64 or uint8 values; an unexpected
format stops inspection rather than falling back to arbitrary unpickling. Sixteen reader
tests passed, including forbidden callable payloads, endian/layout preservation,
legacy protocol3, invalid shapes, nonfinite values, extension-opcode rejection
and trailing payload rejection. Extension references are rejected before native
unpickling because a cached extension could otherwise bypass `find_class`.
Four additional native-panel tests passed, including changed-image and
incomplete-report rejection and recording stereo/depth size mismatch without
resizing it away. Repository Ruff and `git diff --check` passed.
The subsequent complete CPU suite passed782 tests, with the same five existing
dependency/API warnings; no GPU was used. These are software-regression results,
not real registration measurements.

CPU command to run **after** both selective extractions finish:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.audit_ms2_screen \
  --sync-root experiments/ms2_train_screen_sync_01 \
  --depth-root experiments/ms2_train_screen_depth_01 \
  --sync-report experiments/ms2_train_screen_sync_01_report.json \
  --depth-report experiments/ms2_train_screen_depth_01_report.json \
  --plan experiments/ms2_train_screen_plan_01/plan.json \
  --out experiments/ms2_native_screen_01/report.json
```

The sync transfer has now also exited successfully at exactly24,150,818,939
bytes, and its selective extraction was started. Both `.part` files now have
successful terminal transfer results and the expected byte counts; the historical
suffix is retained rather than renaming live extraction inputs. Full bzip2 checks
and complete extraction reports are still pending. The running depth extraction
has produced the16 selected RGB depth maps so far. Do not treat these provisional
members as a complete or integrity-verified dataset.

## Completed depth extraction: actual native measurements

The depth extraction subsequently completed with no missing required files and
32 verified selected PNGs (16RGB plus16thermal). Full bzip2 stream inspection
completed, including CRC/EOF, and the archive hashes before/after matched.
The sync extraction remains live and no cross-camera geometry is claimed yet.

- Archive SHA256: `b148bda7f56ab0370bbb8dfbad05fd1a4b699b083186a4354dc9993e6a60f026`.
- Report: `experiments/ms2_train_screen_depth_01_report.json`.
- Report SHA256: `210bdc9b8a6417215b9a1051eed69352d6499f9768a8d80b521fa77cb450034d`.

Each selected PNG was decoded and its extracted-file hash rechecked. Native
arrays are uint16, RGB384×1224 and thermal256×640. Nonzero coverage below is per
frame; depth medians pool all nonzero samples across the16frames. Values use
the documented division by256, without filling zero-depth holes.

| Camera | Nonzero coverage min / median / max | Depth min / pooled median / max (m) |
|---|---:|---:|
| RGB left | 1.0549% /1.3241% /1.5087% | 2.7773 /12.6016 /117.2500 |
| Thermal left | 4.1302% /5.0354% /5.6873% | 2.7148 /10.0703 /116.4023 |

These measurements confirm **sparse** single-scan support. They do not justify
dense paired supervision, depth densification as GT, cross-camera visibility or
physical registration qualification. They also are not radiometric intensity or
temperature measurements. The selected image/calibration/timestamp extraction
must finish before running the combined native-panel audit above.

## Provisional calibration and depth-convention check

Before the sync stream finished, its2,797-byte `calib.npy` member became available
with SHA256 `6f1303d70ffb4bb168a268294557549a5ec50698bd45ce4a32847732ef7bfd66`.
This identity is provisional until matched against the completed extraction.
The restricted reader initially rejected an actual uint8 dtype in the supplied
identity/zero matrices. Non-executing pickle inspection identified that exact
format; uint8 numeric matrices were then supported with an added regression test.
No matrix values, rotation tolerances or qualification thresholds were changed.
All11 supplied rotation matrices satisfy the existing1e-5 numerical check. Left
rectified-camera rotations are identity, not extra rectification rotations to
apply a second time. This does not prove the cross-camera calibration is accurate.

An exploratory check on the already-frozen endpoint frames compared two explicit
depth conventions. It projected RGB nonzero depth into thermal coordinates using
the supplied transform composition and associated the nearest nonzero thermal
depth pixel within1native pixel. These are nearest-pixel associations, **not known
identities of the same LiDAR return**, and the products may share calibration.

| Frame | Convention | Associated / source points | Median absolute depth residual (m) |
|---|---|---:|---:|
| 000000 | Axial Z | 5,776 /6,203 | 0.01823 |
| 000000 | Euclidean range | 3,345 /6,203 | 0.20520 |
| 010441 | Axial Z | 5,303 /5,304 | 0.00190 |
| 010441 | Euclidean range | 3,042 /5,304 | 0.19528 |

This favors the axial-Z interpretation already suggested by the official
loader's `focal * baseline / depth` disparity conversion. It is not a registration
pass, an independent physical correspondence measurement or a license to select
only the better-looking frame. Conditional residual distributions must be read
with their different association counts, not compared as identical populations.

`src/aero_ir/data/ms2_depth_consistency.py` makes this diagnostic reproducible;
the combined native-panel audit now runs both conventions in both directions on
all16preselected frames after full extraction. Six new tests cover known axial
geometry, the range alternative, absent depth, off-image support and invalid
association radius. The26 calibration/native-panel/depth-consistency tests and
repository Ruff passed after this addition. The next report must retain all
frames and distinguish internal depth-product consistency from actual RGB–thermal
image alignment.

## Completed native audit and ego-motion diagnostic

Both archive extractions have now completed. The sync extraction has124 selected
files, no missing required assets and a successful full-stream CRC/EOF check.
Its archive SHA256 is
`c3a10f2cef0d04ea6999c9885311aa1bda206e3c228960ce3e1462d19e913902`;
`experiments/ms2_train_screen_sync_01_report.json` has SHA256
`78ba19ed835dadcbc2a1c85301d53d02f643b973c67c0868f9db6ef93ed0d157`.
The fully extracted readme matches the earlier suffix preview byte-for-byte.

Actual native audit: `experiments/ms2_native_screen_01/report.json`, SHA256
`9bf321e4d342b2e65654e42814f4f7246e9adee03428fdc94452e3b35f0e9a97`.
It verifies all extracted hashes, the16-frame plan,64 image arrays,32 depth maps,
calibration and timestamps. There are no required calibration omissions,
numerical matrix errors or stereo/depth dimension mismatches. This supersedes
the earlier provisional file-integrity status, not the scientific HOLD.

Across10,442 same-index RGB/thermal records, absolute timestamp skew is
19.5286ms median,52.1377ms p95 and76.3491ms maximum. The16 selected pairs have
20.1687ms median and50.5836ms maximum. Both streams are increasing; this is not
proof that their timestamp definitions equal the physical exposure midpoint.
Neither frames nor timestamps were changed to improve scores.

Static reprojection is therefore compared against an explicit rigid-rig motion
hypothesis using the **RGB odometry trajectory and RGB timestamps only**:

`T_thermal_from_rgb(t1,t2) = T_fixed @ inverse(P_rgb(t2)) @ P_rgb(t1)`.

Do not directly combine separately zeroed RGB and thermal pose streams as if
they shared a world origin. Translation is interpolated linearly and rotation
along its shortest path. Invalid endpoint rotations fail before conversion;
the maximum numerical SO(3) representation adjustment on used valid endpoints
was2.20e-6 per matrix element, recorded rather than hidden. Source poses are
unchanged. This does not repair the previously reported invalid NIR pose.

Fourteen frames use interpolation within recorded poses. Frame000000 needs
12.762299ms extrapolation and010441 needs0.705938ms. The exploratory bounded
variant permits at most20ms, explicitly records these two cases and retains
all16frames. A strict interpolation-only analysis would have14/16 available,
not16/16. No bounded/extrapolated result is promoted to physical GT.

| Direction / variant | Associated / all source depth points | Median of frame-median depth errors (m) | Joint ≤1 target pixel and ≤0.03m / all source points |
|---|---:|---:|---:|
| RGB→thermal, static | 66,810 /98,628 | 0.15064 | 26.54% |
| RGB→thermal, ego-motion | 84,542 /98,628 | 0.00534 | 73.73% |
| RGB→thermal, inverse-motion control | 58,535 /98,628 | 0.24403 | 17.67% |
| Thermal→RGB, static | 28,317 /132,302 | 0.14853 | 10.42% |
| Thermal→RGB, ego-motion | 42,815 /132,302 | 0.00534 | 28.73% |
| Thermal→RGB, inverse-motion control | 23,605 /132,302 | 0.33110 | 6.87% |

The1pixel/0.03m column is an **exploratory reporting bin, not a qualification
gate**; the report also includes0.01/0.1/0.3m bins. Unassociated source points
remain in the joint denominator. Conditional medians have different association
populations, so the report additionally records common-support comparisons.
In frame004872 the forward association count falls from3,901 to1,343 despite a
lower conditional median; this remains visible and is not removed. Forward and
reverse populations differ in native resolution, sparse sampling and field of
view. One native target pixel is not the same angular tolerance in both cameras.

The wrong-direction control worsens both aggregate error and joint support.
This supports ego-motion/timestamp mismatch as a material cause of the **MS²
depth-product** discrepancy. It does not establish the same root cause on
Anti-UAV, prove exact return identities, independently verify camera-image
alignment, compensate independently moving objects or eliminate sparse holes.
The depth products can share calibration, and no dense generator supervision
or research claim is approved from these internal consistency results alone.

Report: `experiments/ms2_ego_motion_probe_01/report.json`, SHA256
`4fb3cb959761172cd12386fc7945a2b729eed499af704c0e035036a9d07f491b`.
Nine motion/intersection-denominator tests passed; repository Ruff and
`git diff --check` passed. Reproduce on CPU with a fresh output:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.probe_ms2_ego_motion \
  --out experiments/ms2_ego_motion_probe_repeat01/report.json
```

That fresh-output replay was actually executed in an independent CPU process.
Its complete JSON report is identical, with the same SHA256 above, not merely
the same rounded table values. The subsequent full CPU suite passed798 tests
with five existing dependency/API warnings. No GPU was launched.

Next: assess actual RGB/thermal image alignment using the measured timing and
calibrated sparse references, preserving unsupported regions and dynamic-object
uncertainty. This candidate does not yet clear registration qualification.
