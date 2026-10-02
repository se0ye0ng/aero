# Completion audit — registration, experiment pipeline and public release

This is a requirements/evidence audit, not a percentage estimate or completion
certificate. Snapshot date:2026-10-02. The objective remains qualified registration,
a completed controlled experimental pipeline, and release through `se0ye0ng/aero`.

| Requirement | Authoritative evidence checked | Current conclusion |
|---|---|---|
| Reproducible real-only detector reference | FLIR baseline `run_manifest.json`, declared inputs/outputs and original Git objects; static verification passed | Baseline available; this check did not rerun its GPU training |
| Registration improves enough to meet the frozen engineering criteria | Completed replay300 comparison:125/160 uniform,124/160 failure-aware midpoints; required95% | Not passed; sampling alone did not resolve the failures |
| Complete controlled300-additional-epoch comparison | `registration_replay_e300_seed0/{uniform,failure_aware}/training_result.json` and verified `comparison.json` | Completed96,000 retained updates per arm; saved-artifact verification passed, not inference replay |
| Verify full eligible train/validation coverage using the new checkpoints | New inference auditor, coverage/coordinate tests and blocked CPU preflight | Code ready; no completed new native-input audit |
| Establish independent physical correspondence on the actual supervision domain | `qualification_v4.gate_report`, existing automatic-matching diagnostics, generator evidence reader | No implemented scientific approval path or accepted independent evidence |
| Validate an explicitly authorized additional RGB–thermal training source without replacing Anti-UAV results | MS² native16-frame audit, motion/depth comparison,96-case image screen and known-resize controls; both archive extractions verified | Internal depth consistency improves, but image-based correspondence still differs by several pixels; source and export are not qualified |
| Export aligned, observed conditioning with captions/masks and label geometry | `DiffV2IRTrainDataset` contract and generator planner | Loader implemented; qualified exporter/conditioning products absent |
| Train a real generator with an approved initialization | Separate upstream runtime and tiny full-path CPU training/resume probe | Hook compatibility repaired; real-data launcher, approved weights and GPU training still absent |
| Compare real, real+simulated and real+generated under the same detector protocol | Real-only FLIR result; `run_grid.py`; general CLI | Full three-arm execution/results incomplete; grid preview is not an execution backend |
| Demonstrate the proposed diagnostic/augmentation benefit | Paired downstream AP deltas, uncertainty/seeds, label audits and planned held-out comparisons | Missing; no benefit claim justified |
| Publish reproducible, licensed results under the correct account | Remote URL and commit email; worktree; result snapshot | Remote targets `se0ye0ng/aero`; changes remain uncommitted/unpushed and release review incomplete |

## Current code and evidence audit

The full CPU suite passes 1,102 tests with five warnings. Ruff code checks and
`git diff --check` pass, but the formatter reports 72 source files needing formatting
and 368 already formatted. All 56 previously unformatted test files were formatted
with identical before/after Python syntax trees; test conditions were not changed.
Consequently `make smoke` is still not passing. Historical
experiment files have not been bulk-formatted, because their byte hashes are
part of retained evidence. The configured Git remote is `git@aero:se0ye0ng/aero.git`
and the local commit email belongs to the requested account; SSH authentication
and an actual push were not tested in this audit. A 510-file first-party source
snapshot was verified before cleanup. Snapshot-based CPU saved-result verification
passed for the Anti-UAV RoMa baseline, its higher-resolution ablation, and the
external development and confirmation panels. This is not neural inference replay
or a complete archive of datasets, weights, dependencies and every historical run.

The [MINIMA-RoMa confirmation](detector_free_matching.md#confirmation-on-additional-images)
provides a real external pixel-error measurement: ungated macro PCK3 is 96.54%
on the first 12 additional same-scene pairs, or 96.35% excluding one pair with
duplicated development labels. A later seven-pair check falls to 85.64%, or 83.48%
after duplicate exclusion, with one large failure (49.63-pixel median error).
Post-hoc cycle/orientation filtering detects much of that failure only by reducing
its retained coverage to 33.87%; it does not repair correspondence. These results
do not supply independent pixel labels on Anti-UAV or authorize generator training.
On Anti-UAV's train16 panel, the frozen confidence gate still passes 0/16; the
ungated box proxy is 5/16 and falls to 4/16 at higher processing resolution.
Some wrong box alignments have excellent cycle closure. Motion-only localization
also remains insufficient: top-three candidate coverage has a post-hoc ceiling
of 12/16, while image-only joint selection preserves both boxes in 11/16.

A trained-coarse-plus-RoMa GPU comparison also fails to repair the train16 panel.
The completed uniform300 checkpoint alone passes the matched-support box proxy
on 11/16 pairs; residual RoMa matching after image-only prewarping reduces this to
4/16 without confidence filtering and 0/16 with the frozen confidence gate.
There are no new passes and seven lost passes in the ungated branch. Native-frame
identities, model/source hashes and saved-field scores were checked. The
same-image known-shift control reaches 910/912 points within three pixels, but
does not establish cross-modal correctness. This combination is not adopted;
the original checkpoint and qualification criteria are unchanged.

The general CLI remains a placeholder and `run_grid.py` only previews schedules.
Existing dedicated detector/registration runners must not be confused with a
completed three-arm execution backend. The generator evidence reader correctly
rejects manually promoted pass strings, but still has no implemented scientific
qualification/export approval path. None of the passing software checks fills
that evidence gap. No method-release or generator-training approval is issued.

## The approval gap is separate from training duration

`src/aero_ir/registration/qualification_v4.py:gate_report()` computes geometric
screen/coverage outcomes, but always returns independent correspondence as
`hold_not_evaluated`, dense paired registration as `hold`, and generator eligibility
as `hold`. This is intentionally an engineering evaluator, **not** an implemented
scientific registration qualifier. Even100% engineering success would not change
those fields. Replacing these strings with `pass` would conceal the missing
measurement rather than solve it.

Likewise `scripts/prepare_generator_training.py` accepts the supported v4
engineering report for consistency checks but returns both scientific authorization
and registered-conditioning-export verification as false. New replay300 audit
reports must not be relabelled as v4 or manually promoted to a pass.

The missing next protocol must establish fine correspondence independently of
the fitted boxes, optimized pseudo-masks and cycle closure, and verify observed
support on the exported supervision grid. Current evidence does not establish
which automated method can provide that. Existing negative diagnostics do not
prove all alternatives impossible. The user-requested automatic repair path is
retained; obtaining manual reviewer labels is not imposed as the next task.

The subsequent [dataset evidence audit](registration_dataset_evidence.md) checked
the archive directory and all320 official-train annotation schemas, not just
filenames. Their only fields are `exist` and `gt_rect`. No independent pixel
correspondence was established. This does not exhaust embedded video metadata or
external calibration records. Missing evidence must not be replaced with invented
calibration or pixel GT.

After explicit user authorization, a separate four-pair UAV-TIRVis landmark
screen was acquired and executed on CPU with frozen XoFTR/LoFTR. It provides an
implemented reference-point error measurement path, not independent evidence on
Anti-UAV itself. The [external diagnostic](registration_external_landmarks.md)
records all pairs, error units, estimator seeds, file hashes and limitations;
neither model is approved as an automatic GT producer from those measurements.
The subsequent [local-warp diagnostic](registration_local_warp_diagnostic.md)
improves external XoFTR PCK3 from36.10% to84.07%, but achieves only1/16 joint
box-corner proxy passes in each tested Anti-UAV header condition. Those distinct
metrics cannot be interchanged. This candidate is not promoted to registration
qualification, and the original downstream/release objective remains incomplete.
The subsequent matched-maximum-budget control-selection diagnostic increased
Anti-UAV forward-fit availability but left joint box-proxy passes at1/16 for
both header conditions. It also lowered external PCK3, so it was not adopted.
Saved-control attribution separates absent fits, unsupported target corners and
incorrect supported alignment; these are not repaired by declaring more fits
successful. Results and reproducible CPU commands are in the local-warp diagnostic.
The [RGB-resolution comparison](registration_rgb_resolution.md) subsequently
decoded the same 16 native pairs and increased only RGB input width from 640 to
1280. Both fixed selection policies retained only the same 1/16 joint box-proxy
pass. Original-input inference replay and known-shift controls passed; the
candidate completed but did not resolve the matching/registration bottleneck.
The [MINIMA-XoFTR checkpoint swap](registration_minima.md) is also complete on
the same external4/Anti-UAV16 panels. The pretrained checkpoint improves external
FPS PCK3 from80.17% to86.41% but leaves Anti-UAV joint box counts unchanged for
both TPS policies and all three predeclared global models. This is another
measured candidate, not an approval path or evidence that every alternative fails.

The user subsequently authorized validation of an additional public paired
generator training source and obtained the MS² authors' download instructions.
The [MS² source audit](registration_ms2_source.md) records a predetermined official
training sequence, separate data licensing, observed odometry anomalies and
verified image/depth/calibration extraction. Its completed native audit found
nonzero cross-camera timestamp skew. An RGB-trajectory ego-motion diagnostic
reduces median-of-frame depth residuals from0.1506m to0.00534m (RGB to thermal),
with increased associated support and an adverse inverse-motion control. Two
endpoint frames require explicitly recorded bounded extrapolation. These are
internal depth-product checks, not independent physical image measurements.
This creates a separate route to evaluate a
calibrated training source; it neither repairs Anti-UAV nor qualifies a generator
yet. The original registration and controlled-experiment requirements remain.

The subsequent [MS² image screen](registration_ms2_image_screen.md) completed
96cases with frozen XoFTR/MINIMA weights. Against the motion-compensated,
depth-consistent sparse reference, forward equal-frame PCK3 is18.47%/18.81%,
with unsupported predictions retained. These are consistency scores against
calibrated depth, not independent image GT. Same-RGB known-resize controls exceed
99.8% forward PCK3, making a gross resize-coordinate bug unlikely on these images.
Depth-free epipolar diagnostics also distinguish correct from unrelated pairs,
but proximity to an epipolar line cannot certify a unique pixel correspondence.
Neither result justifies changing registration/generator authorization flags.

The [RGB-stereo depth follow-up](registration_ms2_stereo_depth.md) then estimates
depth without IR matches or LiDAR inputs. Bounded rotation-only fits using that
higher-support geometry improve own-matcher exploratory check PCK3 from33.96%
to54.21%(XoFTR) and40.82%to64.02%(MINIMA). Those percentages are conditional on
stereo-supported raw matches, not dense TPS-reference accuracy or qualification.
The corrections were frozen for15new frame IDs before their extraction. Completed
confirmation improves own-correction fixed-support PCK3 from21.59%to54.39%(XoFTR)
and30.42%to67.22%(MINIMA), with improvements on15/15frames and poor unrelated-pair
controls. Saved-data scores were recomputed, not just trusted from a report.
This confirms a repeatable correction within the same training sequence, not
physical pixel GT, dense generator support or an Anti-UAV pass. No source
calibration is overwritten. The full registration,
qualified export, controlled downstream comparison and public-release requirements
remain incomplete.

An independent-process confirmation replay reproduced all77output files exactly.
The subsequent [residual attribution](registration_ms2_residuals.md) found that
65.62–72.68% of the remaining own-rotation failed matches cannot reach3pixels by
changing depth alone under the fixed camera model. Four bounded effective
thermal-intrinsic parameters, fitted only on the original8fit frames, increase
own-correction PCK3 on the already observed15-frame panel to61.36%(XoFTR) and
74.95%(MINIMA). This is exploratory reuse, not fresh confirmation. MINIMA has
one regressing frame; neither candidate is qualified or installed. The camera
perturbation may absorb matcher/system errors rather than identify physical
calibration. All840CPU tests passed with five existing dependency warnings.

The [next frozen-camera confirmation](registration_ms2_intrinsic_confirmation.md)
has now completed on15disjoint quarter-interval frames. Own augmented PCK3 is
57.60%(XoFTR) and68.11%(MINIMA), versus49.74%/58.34%for rotation-only. Both improve
all15frames over their own rotation, but each still worsens one frame relative
to the author calibration. Saved-array verification checked751identities.
An image-gradient test without learned correspondences gives small improvements
for polarity-insensitive scores, while signed correlation worsens; this is weak,
mixed corroboration, not independent pixel GT. Missing dense support and large
remaining residuals keep qualification/export on hold. The full CPU suite now
passes858tests with five existing dependency warnings. No GPU job was launched.

The [cross-stereo follow-up](registration_ms2_cross_stereo.md) additionally uses
the right thermal images. Thermal stereo timestamps differ by up to8.12ms, with
modeled motion displacement up to6.04native thermal pixels on the fixed grid.
Time-aware rectification and135actual image-stereo cases increase available depth
coverage but do not consistently improve agreement with RGB stereo depth.
Augmented cameras still lack independent physical validation. All868CPU tests
pass; source/export qualification and the full release goal remain incomplete.

The same 135 cross-stereo cases were then reconstructed in fixed-focal disparity
units, with all 832 recorded identities verified. Author/time-aware discrepancy
is 1.227 equivalent disparity pixels (median of conditional frame medians),
versus 1.308/1.302 for the XoFTR/MINIMA augmented cameras. On the same 10,250
supported points, each augmented camera improves the per-frame median on only
3/15 frames. These are stereo-disparity differences, **not RGB–IR correspondence
errors or physical GT**. The camera candidates therefore remain unqualified.
The full CPU suite passes **872 tests**, with five existing dependency warnings;
Ruff and `git diff --check` pass. No GPU work was launched for this analysis.

A further 90-case thermal preprocessing control exactly reproduces the original
stereo arrays, then compares shared and independent image-percentile windows.
Independent normalization increases supported estimates from 11,281 to 12,697,
but does not improve the aggregate discrepancy on the same 10,664 supported
points (1.173 px original versus 1.177 px independent). All frames and wrong-right
controls are retained. This is exploratory preprocessing sensitivity, not a
registration repair or a qualified source. Full CPU tests now pass **878 tests**
with five existing dependency warnings. No GPU job or generator was launched.

The [same-camera LiDAR/stereo comparison](registration_ms2_lidar_stereo.md)
completed all 96 planned cases on the original 16 frames. Sparse projected
LiDAR adds a range measurement but shares calibration and projection uncertainty.
Time-aware RGB/thermal stereo discrepancies are 0.704/1.094 equivalent thermal
disparity pixels, respectively, with all unsupported LiDAR locations retained
in full-reference fractions. Two out-of-range pose cases per sensor are recorded
as unavailable, not extrapolated or omitted. This prevents adopting thermal
stereo as an already validated dense reference. All 883 CPU tests pass; the
registration, generator-export and release requirements remain incomplete.

An [inference-only RAFT-Stereo comparison](registration_ms2_raft_stereo.md) is now
prepared for user-run GPU execution, retaining the same 96 LiDAR-reference cases.
Pinned official weights passed a real CPU synthetic 8-pixel shift control
(median error 0.03622 px; all 7,475 scoring points within 1 px). The full preflight
verified 421 identities. The full CPU suite passes 891 tests, but **no real-data
GPU comparison result exists yet**. A prepared command and synthetic control are
not a registration repair or generator authorization.

The separate CPU RAFT comparison analyzer is prepared without modifying the GPU
runner or any of its 421 frozen identities. It reconstructs geometry-derived
support and reverse-consistency masks, validates protocol/case labels, and
separates coverage gains from accuracy on common points. Thirteen additional
tests bring the full suite to 904 passing tests. The default GPU output directory
is still absent at this check; no active GPU job or completed result is inferred
from the previously supplied command. Actual results remain required.

## Pipeline and release boundaries

`scripts/run_grid.py` formerly printed commands and exited successfully even when
`--dry-run` was omitted, while its execution branch contained only `pass`. It now
rejects non-dry execution with an explicit error before any launch. The general
`aero_ir.cli` dispatcher also remains unimplemented. Dedicated tested training
wrappers are real runners; neither listing configurations nor a zero shell exit
from the old enumerator establishes completed experiments.

Public summaries separate detection AP from registration pass fractions. They
contain aggregate numbers and hashes, not native imagery, weights, private paths
or reviewer identities. `docs/inspiration.md` is a retained local requirements
file mentioning NDA-governed data and is ignored pending explicit provenance/
redistribution review. It was not deleted. No blanket claim that every local
untracked file is safe to publish is made.
The present single-seed numbers are preliminary engineering records, not the
seed-spread evidence required for a confirmatory claim by `CONTRIBUTING.md`.
The local paper-style summary does not make the repository scientifically ready
for release as a completed method; that policy review remains open.

Current Git configuration uses the `se0ye0ng` noreply email and remote
`git@aero:se0ye0ng/aero.git`. These settings alone do not prove SSH authentication,
GitHub push success, CI success or scientific completion. A reviewed commit/push
and remote CI verification remain separate release actions.

## Next authoritative evidence

The [tiled matching and spatial-support comparison](detector_free_matching.md#tiled-matching-and-spatial-support)
has completed on the fixed Anti-UAV train16 panel. Whole-image and tiled inputs
each pass 1/16 pairs; reciprocal endpoint filtering and all-consensus piecewise
affine interpolation do not increase that count. A post-hoc box audit finds
946/2,801 reciprocal source endpoints inside a drone box landing outside the
other drone box. This is not pixel GT, but prevents interpreting increased match
counts as a repair. The saved GPU geometry was reconstructed on CPU. No further
run of these unchanged candidates is justified by the current evidence.

The [boundary-control confirmation](registration_local_warp_diagnostic.md#boundary-control-confirmation)
is complete on 12 additional pairs from the same four external scenes. PCK3
improves from 63.37% to 67.89%; unavailable references decrease from 107/499 to
66/499. The duplicated development landmark files in Seaside/2 are disclosed,
not silently omitted. This is partial coverage improvement, not qualification.
Anti-UAV transfer remains 1/16 under both header-handling conditions.
The [dense fine-window comparison](detector_free_matching.md#dense-fine-window-comparison)
also remains 1/16 despite 8.6 times as many matches. These completed experiments
do not establish qualified paired exports, generator learning, controlled detector
benefit or readiness to publish the full method as validated.

The [external local-warp geometry audit](registration_local_warp_diagnostic.md#sampled-geometry-of-the-external-local-warps)
is complete. All sampled supported Jacobians have positive determinants, but
XoFTR forward source-grid support spans only 45.33–71.14%; a separately fitted
reverse map also leaves appreciable round-trip error. These observations do not
establish globally valid dense correspondence or justify unmasked generator
supervision. The [640-versus-1280 input-cap comparison](registration_local_warp_diagnostic.md#external-resolution-comparison)
has now completed on an RTX 3090 and all eight saved cases passed CPU verification.
Equal-pair PCK3 decreases from 83.72% to 51.97%, with all 241 references per cap
retained. Seaside's supported median error rises from 1.057 to 51.997 thermal
pixels, so missing coverage alone does not explain the deterioration. The higher
cap is not promoted; neither setting clears independent dense qualification or
Anti-UAV approval. The GPU run is no longer pending.

The [joint camera optimization check](registration_ms2_residuals.md#joint-rotation-and-intrinsic-optimization)
also completed. Relative to the sequential camera fit, quarter-panel PCK3 gains
are only 0.21 percentage points for XoFTR and 0.34 for MINIMA; both fits hit the
vertical principal-point bound. A fresh run reproduced the report exactly, 739
input identities were checked, and 300 existing baseline score rows matched.
This does not justify installing a physical camera correction. **962 tests pass**.
The [reverse-matching observation check](registration_ms2_residuals.md#reverse-matching-consistency)
also completed all 60 cases. One-pixel reciprocal subsets improve conditional
PCK3 only to 58.71%/69.09%, and approximately 74% of wrong-image matches also
remain reciprocal. All original denominators remain reported. Verified artifacts
and independently reconstructed masks/scores do not make reciprocity independent
pixel GT; this route does not clear qualification or justify pseudo-label export.

The [fixed-projection cross-camera check](registration_ms2_raft_stereo.md#fixed-projection-cross-camera-check)
completed 135 cases with exact zero-offset replay. Thermal-only +0.5 px sampling
reduces time-aware common-point equivalent disparity discrepancy from 1.2055 to
1.1621 px, improving only 8/15 frame medians; full-reference within-3-px fraction
increases from 59.98% to 60.87%. Verification checked 1,218 input identities,
135 artifacts, fixed RGB depth/projections and camera geometry, and reconstructed
scores. **957 tests pass**. This is modest stereo-depth improvement, not RGB–IR
registration recovery: the RGB-to-IR pixel mapping itself is unchanged. Further
same-camera training is not justified by this result as a registration solution.
The remaining requirement is validated cross-modal spatial correspondence, not
another reduction of same-surface depth error. All scientific approval remains HOLD.

The [disjoint-frame sampling confirmation](registration_ms2_raft_stereo.md#disjoint-frame-confirmation)
is complete: 30 source depth maps were extracted and verified, then 270
sensor/frame/condition/offset cases were evaluated. On common thermal points,
time-aware median discrepancy decreases from 1.0599 to 0.8708 equivalent pixels
with the fixed +0.5 px right-image sampling offset, improving 12/15 frames and
worsening 3/15. RGB does not benefit on common points. Verification checked 882
input identities, 270 artifacts, all original depth reference coordinates,
pairing, fixed geometry across offsets, rescored maps and aggregates. This is
same-sequence confirmation on frames previously used for other diagnostics, not
an independent scene or dense RGB–IR correspondence qualification. **953 CPU tests
pass** at that stage. The subsequent cross-camera check above leaves RGB stereo
and camera parameters unchanged.

The completed [vertical sampling intervention](registration_ms2_raft_stereo.md#vertical-sampling-intervention)
reproduced the unchanged baseline in all 96 cases and evaluated both half-pixel
directions. Thermal time-aware common-point discrepancy decreases from 1.1332
to 1.0190 equivalent pixels with +0.5 px sampling, improving all 14 available
frames; RGB common-point discrepancy worsens slightly. This is a thermal-specific
candidate for fixed-protocol confirmation on different frames, not a universal
camera correction. All 727 input identities, 288 artifacts and reconstructed
scores were checked. CPU tests: **937 passed**. Qualification and generator
approval remain on hold; no new GPU training is required for this follow-up.

The [alternative-anchor vertical diagnostic](registration_ms2_raft_stereo.md#vertical-profiles-with-alternative-horizontal-anchors)
completed 96 cases. On identical common points, all three anchors retain an
aggregate +0.5 px vertical peak in both sensors. All 176 real-texture known-shift
controls passed, without changing source images or camera parameters. This narrows
the effect beyond LiDAR-specific horizontal anchoring, but does not isolate a
physical calibration cause. Verification covers 724 input identities and all
saved artifacts, zero-offset scores, control results and summaries; **933 tests
pass** at that stage. The subsequent intervention above retains adverse pair
controls and all reference points. No physical camera correction is approved.

The subsequent 96-case [horizontal/vertical profile diagnostic](registration_ms2_raft_stereo.md#horizontal-and-vertical-score-profiles)
is complete. All 14 available RGB/time-aware frames prefer median right-image
vertical offset +0.5 px; thermal prefers +0.5 px in 10 frames and +1 px in four.
This is a bounded, LiDAR-anchored photometric finding, not an applied or qualified
calibration correction. Unrelated pairs can have larger local peak gaps, so
uniqueness within this short axis search does not establish physical truth.
All 624 input identities, 96 saved artifacts, zero-offset controls and aggregate
scores were verified; **927 tests pass**. The next safe test replaces the horizontal
LiDAR anchor with SGBM/RAFT anchors to distinguish vertical residual from anchor
coupling, before fitting or installing a correction. No additional GPU is needed
for that diagnostic. Registration, qualified export and release remain incomplete.

The 2026-10-01 CPU [image-patch follow-up](registration_ms2_raft_stereo.md#image-patch-evidence)
completed all 96 cases with fixed LiDAR/SGBM/RAFT disparities and retained all
reference points. Time-aware thermal common-patch NCC medians are
0.86901/0.92450/0.91318, respectively, but seven unrelated-right thermal matches
also give high SGBM/RAFT NCC. Thus patch correlation is not an automatic physical
GT producer. The calculation passed 524 input identities, 96 artifact/reference
and score checks, and 8,448 independent SciPy point/method interpolation checks.
All **920 CPU tests pass**, with five existing warnings. No GPU work, camera
correction, reference filtering or approval occurred. The next investigation is
horizontal/vertical score sensitivity and ambiguity, not another blind model swap
or promotion of either estimator's output to truth. Full release and the original
Anti-UAV registration requirement remain incomplete.

Update after the user identified node31: the RAFT GPU report appeared and all
96 cases passed the separate CPU analyzer, including 521 provenance/artifact
identities and reconstructed support masks. The
[completed comparison](registration_ms2_raft_stereo.md) increases thermal/time-aware
support from 55,005 to 102,980 points, but worsens common-point median discrepancy
from 1.057 to 1.225 equivalent pixels (4/14 frame medians improve). The missing-GPU-
result blocker below is therefore superseded; lack of qualified physical evidence
is not. No additional GPU training is launched or generator approved. The next
scientific question is the source of image-stereo versus projected-LiDAR bias,
not whether this inference job completed.

The 2026-09-25 public-document review additionally ran the missing formatter
check. `ruff format --check src tests scripts` fails with **127 files needing
formatting and 213 already formatted**. Thus Ruff code checks and 912 passing
tests do not imply that `make smoke` passes: its `lint` prerequisite also checks
formatting. README now separates this current release blocker from the historical
92-test smoke record. No bulk formatting was applied to frozen experiment source,
and no manifest was rewritten. Formatting/release cleanup must preserve exact
experiment snapshots before changing hashed files. The paper-style results page
now distinguishes additional-source MS² diagnostics from detector and Anti-UAV
measurements, without claiming a missing GPU result or qualification.

On 2026-09-25 the default RAFT output was still absent; no live inference job is
inferred from an allocation or an earlier execution intention. Its GPU command
has been supplied to the user. Meanwhile a separate
[LiDAR-reference sensitivity analysis](registration_ms2_lidar_reference.md)
reconstructed all 96 baseline cases and reproduced its report byte-for-byte in
another CPU process. Thermal one-depth-unit perturbations change modeled disparity
by at most 0.0632 px; low-local-variation groups still have about 1.05 px residual.
This weakens encoding precision and boundary-only explanations, without assigning
a unique cause or certifying LiDAR as dense GT. All 421 frozen RAFT inputs remain
unchanged. Full CPU suite: **912 passed**, five existing warnings; Ruff and diff
checks pass. GPU comparison, registration authorization and public release remain
incomplete.

The two GPU jobs have completed and their comparison has been verified. Neither
passes the train prerequisite, so their completion is not a reason to launch
the native-input validation audit or a generator. CPU attribution on the hashed
midpoint inputs identifies remaining box-position/extent failures across small
and larger targets, not cycle-error magnitude violations. A repair needs a new,
controlled test of that alignment failure, with passing-case retention measured.
If a candidate later passes the existing train rule, execute the frozen
native-input engineering audit; it still cannot erase the physical/export gap.
No goal completion or unconditional generator command is justified at this snapshot.
