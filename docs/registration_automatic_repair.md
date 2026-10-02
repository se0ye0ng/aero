# Automatic registration repair: executed evidence and next GPU probe

**Current action:** SAM v1 has completed. Its crop-centering/aspect-ratio and
search-range limitations were diagnosed and repaired in a separate implementation.
CPU cached-mask replay and fresh SAM v2 GPU extraction are complete; see the
[SAM v2 corrected-crop results](registration_sam_v2.md). The subsequent
[CPU non-deterioration ablation](registration_sam_pareto.md) prevents boundary-score
regressions, but cannot distinguish physical alignment from optimized pseudo-masks.
No repeat GPU job is needed to reproduce that CPU comparison.
The [shared-velocity CPU initialization check](registration_shared_velocity.md)
also completed:8/16 geometric passes, improved cycles but worse box alignment.
Naive reuse of a displacement as a velocity is not accepted as the repair.
The [v7 shared-velocity learning pilot](registration_v7.md) now jointly optimizes
alignment and reciprocal geometry, with matched geometry/control and MIND arms.
CPU training smoke and command preflights passed; GPU results must be verified
before any further training or generator decision.
The original launch instructions below remain a record of v1, not the current job.

## Correction of direction

Human annotation is **not** a prerequisite to test a repair loss, automatic mask,
input ablation, timing hypothesis or invertible transform. The old48-pair and
six-pair forms are paused. Existing A responses remain unmodified exploratory
data; free-choice landmarks do not directly measure inter-reviewer agreement.

Not all proposed alternatives were previously tried. MIND, MI and NGCC results
were bounded frozen-model candidate searches, not loss-augmented training.
Whole-image DINOv2 did not test ROI descriptors or feature-loss training.
At that time SAM masks, TPS/local invertible training and temporal aggregation
had not been run. SAM v1 is now complete; the latter two remain unexecuted.
Do not turn these limited negative results into claims that the methods cannot work.

The deployment/paired-generator qualification criteria remain unchanged. Automatic
proxies are useful for selecting repair hypotheses, but cannot be renamed pixel GT.
No reviewer response is necessary for the experiments below.

## Executed: native ROI + automatic HUD ablation

Report: `experiments/registration_auto_roi_train16_01/report.json`.
CPU execution:196.34 seconds after model initialization. Same16 train midpoint
pairs and pinned XoFTR weights; native decoded-frame and annotation hashes checked.
No validation/test access and no checkpoint updates.

Each modality uses its existing **train GT box** to create a square3x context crop,
resized to256px. Thus this is human-free but **not annotation-free**. Both conditions
discard the same heuristic HUD regions and8px model-grid guard band. Central
thin extreme-intensity axis-aligned lines are detected automatically; the header20%
is also excluded. The detector can miss HUD or remove real scene edges. Inpainted
pixels never count as correspondences; filling can still affect model context.

| Input condition | Actual pairs: >=4 unique reciprocal target pairs | Shuffled sequence controls: same statistic | Actual / shuffled also >=10% hull in both boxes |
|---|---:|---:|---:|
| Native ROI, raw input | 8/16 | 9/16 | 6/16 vs4/16 |
| Native ROI, automatic HUD inpainting | 8/16 | 10/16 | 6/16 vs4/16 |

Four known(+8,-8) crop-grid shifts yielded100% of interior matches within3 model
pixels (709,725,725,717 matches). This verifies implementation on same-modality
controls, **not cross-modal correctness**. Shuffled pairs contain UAVs too and can
share semantic structure, but are not the same frame/physical observation. The
correspondence counts fail to clearly distinguish the actual pairs from these hard
controls. They do not support promoting automatic matches to pseudo-GT as-is.

Crop-normalized box IoU is deliberately not reported as success: box-centred
cropping already supplies coarse alignment. This was a native-ROI and input-HUD
experiment, not training XoFTR or an evaluation of dense full-image registration.

Optional GPU reproduction at256 and512px (not the highest-priority next experiment):

```bash
bash scripts/run_registration_auto_roi.sh
```

The512px condition is prepared, not yet executed. Larger crops do not manufacture
new native-image detail. Outputs refuse overwrite; set `AERO_AUTO_ROI_OUTPUT` for
a fresh repeat. Scheduler GPU visibility is preserved.

## Executed: automatic frame-lag hypothesis screen

Report: `experiments/registration_timing_train16_01.json`. Existing RGB/IR train
box-centre trajectories are compared over lags−15…+15. Four chronological blocks
each fit a2D affine trajectory map using their first2/3 and check the last1/3.
All lags use identical common existence support. The selected lag minimizes only
fit error; check errors are reported separately. A nonlinear synthetic+4-frame
control recovered+4 in all four blocks.

Only `20190925_210802_1_7` selected the same nonboundary nonzero lag in all four
blocks: RGB[t] versus IR[t+2], with check-RMSE decreases64.6%,73.1%,56.8%,40.4%.
This is a **timing hypothesis**, not verified camera calibration. Other sequences
often selected different lags in different blocks. Independent camera tracking,
annotation noise, parallax and affine approximation can confound this score.
No dataset pairings were changed and there is no justified dataset-wide lag fix.

### Full v7 train-cohort timing check

`scripts/probe_registration_timing_cohort.py` extends the same fixed lag search
to all160 official train sequences and joins the outcome to the completed v7
geometry midpoint screen. It adds a cyclic next-sequence IR trajectory control
without changing the fit/check rule or searching validation/test data. Every
annotation hash agrees with the original training-cache shard. Native video
container dimensions are read, but no video frames are decoded or used to fit a
registration model. Adjacent sequence IDs can share a recording, so the shuffled
control is not guaranteed to be an unrelated motion trajectory.

Actual completed report: `experiments/registration_timing_train160_v7_02.json`.
The earlier `_01` has the same results; `_02` was rerun after source line wrapping
and a limitations-string clarification, and binds the current source hashes.

| Completed v7 geometry train-midpoint group | Sequences | Actual-pair stable nonzero lag hypotheses | Shuffled-control hypotheses |
|---|---:|---:|---:|
| Failed joint geometry criteria | 46 | 0 | 0 |
| Passed joint geometry criteria | 114 | 1 | 0 |

All46 failing sequences had enough common frames to run the screen. Among114
passing sequences, one actual pair and two shuffled controls had insufficient
common support; they remain in the table's denominator rather than being dropped.
The only stable actual hypothesis is the previously identified
`20190925_210802_1_7`, at+2 IR frames. There is no new evidence for a global lag
correction or for this lag heuristic explaining the46 remaining v7 failures.

This is **not** proof that the videos are synchronized. Moving-camera effects,
annotation noise, time-varying lag and the affine trajectory approximation can
hide real timing errors. The comparison also relates whole-sequence trajectories
to one midpoint's geometric outcome, not a causal relationship or physical GT.
The dataset's pairing, checkpoints and qualification criteria remain unchanged.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_timing_cohort \
  --out experiments/registration_timing_train160_v7_repeat01.json
```

Use a fresh filename. This is a CPU diagnostic and not a replacement for the
pending matched residual-learning GPU pilots or independent physical evidence.

## Completed initial GPU experiment: automatic SAM silhouettes + shared inverse

Implementation: `scripts/probe_registration_sam.py`; launch:

```bash
AERO_ANTIUAV300_ROOT="$AERO_DATA_ROOT/Anti-UAV300" \
  bash scripts/run_registration_sam_probe.sh
```

The wrapper checks CUDA, installs only `segment-anything==1.0` with`--no-deps`,
runs CPU regression tests, and downloads the official SAM ViT-B checkpoint
(~375MB) if absent. The checkpoint SHA256 and installed package source hashes are
recorded in each result; the first download relies on the official HTTPS origin,
not an independently pre-published SHA256. It uses existing train boxes as prompts,
not manual points. [Official SAM documentation](https://github.com/facebookresearch/segment-anything)
describes box-prompted masks and provides the checkpoint.

Fixed protocol,16 train pairs:

1. Native target context crops, the same automatic HUD exclusion, grayscale RGB
   repeated to3 channels and grayscale IR repeated to3 channels.
2. SAM masks from the original prompt, expanded prompt and translated prompt.
   Reject empty/tiny masks, crop-boundary/HUD contact, >50% context fill, or prompt
   jitter IoU<0.8. These are diagnostic heuristics, not calibrated accuracy criteria.
3. Compare225 predeclared orientation-preserving similarities (scale0.9/1/1.1,
   angle−10/0/+10 degrees, x/y translations−8/−4/0/+4/+8 crop pixels).
   Select using symmetric Dice loss + symmetric Chamfer/256 from original prompts.
4. Use the **same transformation's analytic inverse** for the reverse direction;
   reject foreground lost to invalid support. Evaluate the fixed selected transform
   on perturbed-prompt masks without refitting. These masks are correlated outputs,
   not independent GT. Run the same search on cyclic shuffled-sequence pairs.
5. Save masks, crop metadata, transformation/inverse, exclusions, all per-pair
   results and known-shift controls. No human input or approval required.

This is a **mask-driven alignment pilot**, not300-epoch registration training and
not SAM fine-tuning. It tests an untried automatic anchor before using it as a
training loss. It is not a frozen-v6 checkpoint comparison. Better optimized Dice
alone is expected by construction and not a success criterion. Assess coverage,
prompt stability, paired-versus-shuffled separation and perturbation checks jointly.
Only then consider a matched-budget train-only loss/architecture experiment and
protocol-locked held-out evaluation. Physical same-surface pixel accuracy, temporal
calibration and full-image paired-supervision eligibility remain unestablished.

GPU output: `experiments/registration_sam_train16_v1/report.json` plus32 mask NPZs.
Choose a fresh `AERO_SAM_OUTPUT` if this directory exists. No old experiment,
reviewer response, dataset, checkpoint or qualification threshold is overwritten.
