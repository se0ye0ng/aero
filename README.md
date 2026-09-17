# AERO — When Does Generated Infrared Data Help Detection?

**A radiometric-consistency study of generative training data for infrared target detection.**

> Published downstream effects vary with dataset, task and protocol. This repository tests a
> narrower question: whether sensor-aware radiometric statistics add held-out predictive value
> for infrared detector training data.
>
> Every claim about prior results is anchored to a citation in
> [`docs/references.md`](docs/references.md). Nothing in this repository derives from
> non-public material — see [Provenance](#provenance).

**Central hypothesis.** Across held-out generators and infrared domains, a class-conditional,
sensor-aware radiometric diagnostic predicts the downstream change in detection AP better than
perceptual metrics (FID / LPIPS / SSIM) and recent detection-data metrics (SDQM / CCDM). This is
a hypothesis to test, not an assumed property of RFS.

---

## Contents

- [Why this repository exists](#why-this-repository-exists)
- [Phases](#phases) — the implementation plan, phase by phase
- [Current GO / NO-GO gate](#current-go--no-go-gate)
- [Moving this repository to a GPU machine](#moving-this-repository-to-a-gpu-machine)
- [Compute budget](#compute-budget)
- [The pipeline](#the-pipeline)
- [What the controlled protocol fixes](#what-the-controlled-protocol-fixes)
- [Repository layout](#repository-layout)
- [Mapping to industry requirements](#mapping-to-industry-requirements)
- [Data](#data)
- [Reproducibility](#reproducibility)

---

## Why this repository exists

Real infrared imagery is expensive to acquire: it is bounded by physics (range, aspect,
atmosphere) and by time (flight hours, range slots). Every operational IR detection program
therefore falls back on simulated or generated imagery, and every one of them runs into the
same question — *can this data be trusted for training, and under what conditions?*

Published downstream effects vary with the dataset, task and protocol. Recent detection-specific
metrics and task-oriented IR generation also make the broad version of this question occupied
territory. What remains unresolved is whether sensor-aware radiometric statistics add predictive
value across held-out generators and infrared domains while the training protocol is controlled.
That narrower question is what this repository tests. See
[`docs/references.md`](docs/references.md) for the specific positions and measurements.

---

## Phases

Each phase states what to implement, how to run it, and **the condition that must hold before
moving on**. The exit criteria are not formalities: Phase 1 in particular gates everything
after it, because an unqualified baseline cannot support later interpretation.

| Phase | Goal | State | Runs |
|---|---|---|---|
| [0](#phase-0--scaffold-and-instrumentation) | Scaffold, sensor chain, RFS diagnostic | **CPU and RTX 4090 CUDA pilots passed** | 1 data-free pilot |
| [1](#phase-1--data-layer-and-the-e1-protocol-transfer) | Data layer, detector, **three-arm protocol transfer** | deterministic v2 baseline and replay passed; paired training source on HOLD | 39 screening |
| [2](#phase-2--the-controlled-experiment-f1-f3) | Pretraining and budget ablations — **the sign-flip test** | | 87 screening |
| [3](#phase-3--label-audit-and-failure-mode-decomposition-f4-f5) | Label audit, stratified analysis | | 0 |
| [4](#phase-4--rfs-predictive-power-n1) | Curation, **RFS vs task-aware metric predictive power** | | 30 screening + confirmation |
| [5](#phase-5--small-target-external-validation-and-sensor-matching) | Sequence-disjoint small-target external validation | 410 external adapter passed; 300 global registration failed | 30 screening + confirmation |
| [6](#phase-6--sensor-in-the-loop-generation-n2) | Differentiable sensor in the generation loop | | ~20 |
| [7](#phase-7--deployment-track) | ONNX / INT8 / latency-vs-mAP | | 0 |
| [8](#phase-8--radiometrically-consistent-3d-generation-n3) | 3D multi-view IR generation | | TBD |
| [9](#phase-9--outputs) | Preprint, figures, cards | | 0 |
| [E6](#optional--e6-capacity) | *optional* — does capacity change the effect? | | 36 |

**The three arms.** Every mixing experiment compares **real**, **real + simulated**, and
**real + generated**. The middle arm is `synthetic_baseline` — a deterministic, unlearned
pseudo-IR renderer (`src/aero_ir/generate/synthetic_baseline.py`). Without it, a gain over
real-only could be a gain any crude simulation would also produce, and a loss could be a loss
any non-real imagery would produce. **The generative model is only interesting to the extent
it beats the free option.**

### Current GO / NO-GO gate

As of 2026-09-16:

- **GO:** `make smoke` passes with 92 tests. The data-free pilot separated faithful RFS (0.542)
  from degraded RFS (105.192), flagged inverted polarity as infinite mismatch, and
  back-propagated a finite gradient through the sensor chain on CPU and CUDA. On an NVIDIA
  GeForce RTX 4090 with `torch==2.9.0+cu128`, the 10-iteration CUDA fixture measured 1.158 ms
  per forward/backward iteration and 43.4 MiB peak allocated CUDA memory. This micro-pilot
  validates execution and differentiability; it is not a detector-training runtime estimate.
- **GO:** the local FLIR v2 audit found 10,742/1,144/3,749 thermal train/val/test images, a
  corresponding 16-bit TIFF for every image, zero invalid boxes or missing annotation targets,
  no video-id overlap among splits, and 3,749 valid time-synchronised video pairs. The 32-image
  16-bit RFS run completed; its small pooled sample is diagnostic only. One RGB-train JPEG is
  absent from COCO and is excluded as an orphan rather than silently entering a run.
- **GO:** the FLIR train/validation detector inputs are frozen in a content-addressed manifest
  (`1a730bf3...c638b0`), with the six declared classes mapped explicitly. A deterministic
  512-image, train-only histogram fit freezes the analytics16 network window at 6076-8097 DN
  (`3763f8e6...e19184`). The lazy registry loader, filtered COCO export, COCO evaluator and
  pinned YOLOX 0.3.0 CPU model/data preflight pass. Deterministic 128/64-image train/validation
  subsets are engineering-smoke inputs only and must never be reported as an experiment result.
- **GO (test-pair provenance):** all 3,749 official time-synchronised `video_test` RGB/thermal
  pairs across eight sequence pairs are frozen in an immutable manifest
  (`94d1a3b1...57acd69`). Every mapped image is present in COCO, every frame index agrees, every
  display/analytics file exists, and the map is one-to-one. This manifest is restricted to
  post-freeze generator evaluation and registration diagnostics; it is not a training source.
- **GO (engineering only):** the one-epoch, real-only YOLOX-s GPU smoke completed on `node39`
  with batch 8 and FP16. It trained all 16 iterations, wrote three checkpoints, and completed
  validation in 122.981 seconds wall time. The logged CUDA allocation was 1,812 MiB; at the
  last reported training iteration the total loss was 17.4, and evaluation measured 38.14 ms
  forward plus 4.44 ms NMS. The content-addressed smoke record is
  `experiments/flir_yolox_smoke_result.json` (`be11b7a1...7e9f1b9`). Its best AP of 0.00 after
  one scratch epoch on 128 images is expected and is **not** a scientific result.
- **GO (engineering only):** the batch-8 x 8-step accumulation smoke also passed on `node39`.
  It executed 16 microbatches as exactly two effective-batch-64 optimiser steps, kept losses
  finite, completed validation, and recorded 1,812 MiB allocated CUDA memory. Its successful
  attempt took 46.909 seconds, but its asynchronous per-iteration log is not a valid throughput
  benchmark. The content-addressed record is `experiments/flir_yolox_accum_smoke_result.json`
  (`fe652608...c849237`); its 128/64-image subsets and AP remain engineering evidence only.
- **GO (runner):** the verified FLIR runner now freezes full train/validation input
  hashes, code hashes, resolved configuration and environment; rejects subset annotations for a
  formal run; writes isolated replayable manifests; and persists predictions plus the complete
  COCO AP/AR and per-class metric set. Its bounded timing mode retains the normal 300-epoch
  augmentation state, synchronises CUDA around measured iterations, excludes warm-up, and skips
  checkpoints/evaluation. Formal training specs are prepared only from a clean committed
  checkout; the recorded `git.dirty` field remains part of the review gate.
- **GO (full-data timing):** the selected batch-32 x 2-step accumulation policy ran 160
  microbatches as 80 optimiser steps over the full FLIR loader, visited six multiscale input
  sizes, retained mosaic/mixup and kept every loss finite. It used 9,827 MiB peak allocated CUDA
  memory and measured 19.21 images/s including one-time cold starts for previously unseen input
  sizes. Reusing 672 px returned to about 0.20 seconds per microbatch. The verified manifest is
  `experiments/yolox_runs/flir_real_only_timing_b32_a2_w8_seed0_v1/run_manifest.json`
  (`91da0e27...601f0ac`). This engineering run truthfully records the then-dirty predecessor; it
  must not be rewritten as a clean run.
- **RECORD (v1 baseline candidate):** the clean-snapshot, real-only YOLOX-s baseline completed
  all 300 epochs in 5.987 hours on `node39`, using 11,104 MiB peak allocated CUDA memory. On all
  1,144 validation images it reached mAP@0.5:0.95 0.3513, mAP@0.5 0.5757, mAR@0.5:0.95 0.4683 and
  mAR@0.5 0.7480. Epoch 299 was best at 0.35134 mAP@0.5:0.95; epoch 300 was effectively equal at
  0.35131. The manifest (`6eda51cd...70a4f6`), inputs, predictions, complete per-class metrics and
  checkpoints verify against clean Git commit `fc51f66...775220`. L1 box regression is disabled
  by YOLOX during mosaic training and correctly becomes nonzero in the final no-augmentation
  phase, beginning at displayed epoch 285.
- **HOLD (v1 replay):** the 2026-09-07 full replay completed, but correctly failed the frozen
  0.002 metric tolerance. It reached mAP@0.5:0.95 0.35744 versus 0.35131 originally
  (`+0.00613`), mAP@0.5 0.58947 versus 0.57572 (`+0.01375`), and produced 61,680 versus 59,048
  detections. Sampler, optimiser-step and multiscale-size schedules were identical, while losses
  differed from the first logged interval. The cause is YOLOX 0.3.0's
  `worker_init_reset_seed`, which uses `uuid.uuid4()` for mosaic/mixup/flip augmentation workers.
  The failed replay log is retained locally with SHA-256 `3a8e6bc5...66e6eed`; v1 remains frozen
  and must not be rewritten or rescued by weakening the threshold after seeing the result.
- **GO (determinism engineering gate):** two independent 20-epoch, seed-0 GPU runs produced
  byte-identical final checkpoints (`db76237e...50333e`) and predictions
  (`0707d2a3...0987db`), identical scientific metrics, and identical normalised loss/LR traces
  across all 20 logged epochs. Both runtime records confirm deterministic Torch algorithms,
  deterministic cuDNN, disabled cuDNN benchmarking, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and
  `PYTHONHASHSEED=0` on the RTX 4090. The content-addressed comparison report
  `experiments/flir_yolox_determinism_smoke.json` (`11fdf3de...9d48c`) passes. Early FP16
  checkpoints produced zero-extent boxes; the evaluator now excludes and counts only such
  non-representable outputs at the model-to-COCO boundary. The final evaluation excluded none.
- **GO (v2 baseline reproducibility):** the clean deterministic 300-epoch baseline completed in
  6.66 hours on `node39`. On all 1,144 validation images it reached mAP@0.5:0.95 0.35565,
  mAP@0.5 0.57953, mAR@0.5:0.95 0.47464 and mAR@0.5 0.74193 from 57,114 detections; the final
  evaluator excluded zero boxes. The manifest (`3fd81242...196ea`) and every recorded artifact
  pass static verification. A full 300-epoch replay completed with no failures anywhere in the
  complete nested metric tree at the frozen 0.002 tolerance (`replayed: true`, `ok: true`). The
  replay used the directly matching checkout (`verified_from_git` was empty). Its exact clean
  source commit `6f9b7dc3...ab7915` remains retained by tag
  `flir-v2-baseline-source-6f9b7dc`.
- **RECORD:** the local mirror archive contains 11,886 thermal still images, whereas FLIR's page
  and bundled README state 9,711. The internally consistent local release may be used only under
  its recorded archive hash and counts; comparisons must not call it an unspecified "FLIR v2".
- **GO:** the Anti-UAV300 archive passes CRC and layout checks under its recorded SHA-256. The
  extracted 160/67/91 train/validation/test sequences exactly match the mutually disjoint
  supplied manifests. All 636 paired videos open at 20 FPS; label, visible-video and IR-video
  frame counts agree. The train-only 48-sequence RFS diagnostic completed without reading
  validation/test samples (reference 1.114; internal holdout 1.095).
- **HOLD:** Anti-UAV300 annotation and registration use. Across both modalities, 445 frames are
  marked present but have zero-area boxes (294 IR, 151 visible); adapters must exclude and count
  them. Visible video is 1920x1080, IR is 640x512, modality-presence labels disagree on 14,547
  paired frames, and normalised box-centre residual p95 is 0.160-0.181 by split. Direct visible
  box reuse in IR is invalid until a calibrated transform and residual threshold are frozen.
  `test-dev` duplicates 100 training sequences and is not an independent evaluation split.
- **HOLD (Anti-UAV300 calibrated transfer):** a sequence-balanced robust affine-center and
  log-linear-size transform was fit on all 141,816 usable training pairs, then applied unchanged
  to 57,982 validation pairs. Only 8.82% of training frames and 14.63% of validation frames pass
  the frozen IoU >= 0.6, centroid shift <= 0.25 target diagonals, area-change <= 0.5 and in-bounds
  criteria jointly, far below the required 95%. Validation median IoU is 0.293 and median shift
  is 0.312 target diagonals. The content-addressed audit (`d9575809...5868`) therefore keeps
  calibrated target-box transfer, dense image registration and generator training on HOLD.
  Test metrics are report-only and did not fit, select or qualify the transform. Thresholds must
  not be relaxed after observing this result.
- **HOLD (Anti-UAV300 dense registration):** the failed global-transform
  assumption has been replaced by a device-safe adapter for SuperFusion's image-conditioned
  DenseMatcher. The adapter loads the pinned public RoadScene checkpoint
  (`09337ddd...f9a416`) exactly, removes upstream hard-coded GPU selection and excludes unrelated
  CUDA-only fusion code. A real paired-frame CPU fixture confirms finite dense flow and sub-pixel
  inversion, but scores only 0.561 IoU and therefore correctly fails the 0.6 box threshold; it is
  execution evidence, not qualification. The corrective run fine-tunes on 16 frozen pairs from
  each of the 160 official training sequences for 300 epochs: batch 16 gives exactly 160 updates
  per epoch and 48,000 total updates. It never opens validation/test during fitting. Only then
  did `scripts/run_antiuav300_registration.sh` run a sequence-balanced train/validation screen
  under frozen geometry, flow-validity and edge-alignment thresholds. The 300-epoch checkpoint
  (`a4c8aafe...2388a`) was produced, but the immutable screen (`3dcdb5de...0b7f78`) passed
  only 74.22% of 1,280 training pairs and 67.54% of 536 held-out validation pairs jointly,
  against the frozen 95% requirement. Validation median box IoU is 0.699; its area-change gate
  passes only 76.87% of frames. The runner correctly stopped before the exhaustive 199,798-pair
  audit. This is a qualification failure, not a CUDA or checkpoint-loading error. The fixed
  16-pair-per-sequence cache and mean perimeter/edge loss did not produce box-transfer precision
  sufficient for the frozen gate; repeating the same 300 epochs cannot fix that observation.
  Dense registration and generator training remain **HOLD**. The original checkpoint, report
  and thresholds must not be overwritten or relaxed after observing validation.
- **HOLD (Anti-UAV300 dense registration v2):** the distinct geometry-first protocol was
  committed before execution, initialized from immutable v1, and completed all 300 epochs and
  48,000 optimizer steps over rotating coverage of all 141,816 usable training pairs. Its final
  checkpoint is `13dd5c47...1135f`; fitting opened neither validation nor test. The immutable
  screen (`9e7727a1...15f4c`) then passed only 29.30% of 1,280 training pairs and 22.57% of 536
  held-out validation pairs jointly, versus the frozen 95% requirement. Validation edge evidence
  improved (median edge-NCC gain 0.593; 99.81% of frames improved), but geometry did not: median
  box IoU was 0.386, median absolute area-ratio change was 1.357, and the inverse-residual gate
  passed only 63.25%. The runner therefore correctly blocked the 199,798-pair exhaustive audit.
  This is a scientific qualification failure, not a CUDA or runner failure. The screen,
  checkpoint and thresholds are immutable; paired generator training remains **HOLD**.
- **HOLD (v3 coordinate defect; distinct v4 correction):** the direction audit measured 94.30%/
  92.16% train/validation native-box agreement versus 29.30%/22.57% inverse-box agreement.
  These are different geometric tests, not a corrected estimate of the same registration
  accuracy. Better numerical inversion barely changed the inverse-box pass rate. In addition,
  v3 mixed endpoint image grids with pixel-centre point/cycle calculations. Historical v1--v3
  code and artifacts are retained, but their gates must not authorize paired generator training.
  The separate v4 path converts raw matcher fields once to pixel-centre displacements and uses
  consistent image/point/box/cycle/Jacobian operations. Its engineering gate requires same-frame
  bidirectional and target-local checks; even a pass cannot replace independent correspondence
  evidence. **Generator training remains HOLD.** See [v4 correction and commands](docs/registration_v4.md).
- **GO (Anti-UAV300 native-IR detector engineering smoke only):** the registration-independent
  adapter selects an endpoint-inclusive uniform grid before inspecting labels: eight frames from
  each of 160 training sequences and four from each of 67 validation sequences. It retains
  1,279/266 train/validation images with 1,266/257 boxes and 13/9 true negatives, while excluding
  and counting 1/2 selected zero-extent positives. All decoded 640x512 uint8 PNGs, source-label
  selections and COCO exports verify under manifest `c50c5df6...2f86fa`; the pinned YOLOX-s CPU
  model/data preflight passes with 8,937,682 parameters. This is pipeline validation only. It
  neither accesses test data nor clears the failed paired registration and generator gates.
- **GO (Anti-UAV300 native-IR schedule diagnosis):** the one-epoch GPU smoke passed, and a
  20-epoch diagnostic reached final mAP@0.5:0.95 0.46225, mAP@0.5 0.94987,
  mAR@0.5:0.95 0.53385 and mAR@0.5 0.96498 on the frozen 266-image validation subset. That
  diagnostic is not a standard-training result: the former smoke-derived configuration disabled
  warmup and placed every epoch in YOLOX's final no-augmentation phase. Explicit profiles now
  prevent that error. `smoke` permits exactly one epoch; `standard` requires more than 20 epochs
  and fixes five warmup plus 15 final no-augmentation epochs. For the frozen 1,279-image training
  subset, the guarded 300-epoch configuration resolves to 6,000 optimizer updates. Its full GPU
  result is pending and remains a native-IR subset experiment, not the blocked paired E5 study.
- **GO (Anti-UAV410 external evaluation only):** the 9,361,681,896-byte source archive is
  CRC-clean under SHA-256 `339e0e56...e055`. Its 200/90/120 train/validation/test sequences are
  mutually disjoint and contain 213,995/94,711/129,691 frames; one sampled header from every
  sequence is RGB-encoded 640x512 thermal JPEG. The adapter freezes only the official test split
  in manifest `d5e391ff...0bf9d`, preserves sequence id, frame index, visibility, attributes and
  target-area bin, retains all 2,622 negative frames, and excludes 64 invalid positive frames
  (41 non-positive boxes, 23 out of bounds). The resulting COCO export contains 129,627 images
  and 127,005 boxes (`55b3edfc...0ce3`). Attribute completeness remains HOLD because arrays are
  absent in 90/120 test sequences. Anti-UAV410 remains prohibited for training, generation and
  curation; this gate does not clear the Anti-UAV300 registration hold.
- **HOLD (FLIR paired training and direct label transfer):** the provider supplies an official
  map only for `video_test`, not for RGB/thermal training or validation stills. Shared unique
  `(track_id, category_id)` keys provide 676 box comparisons, but all occur in only one of eight
  mapped sequences; the other seven have none. Even in that sequence, normalised box-centre
  residual p95 is 0.0638 against the frozen 0.02 limit (median normalised IoU 0.5852). Cross-modal
  track-id semantics are undocumented. The registration audit (`6b6602eb...64bd8e9`) therefore
  prohibits guessed still-image pairing, direct box reuse, transform calibration from test data,
  and generator training on these pairs.
- **HOLD (three-arm publication):** the real-only v1 baseline and replay are plausible independent
  outcomes, but the replay failed its frozen reproducibility tolerance because augmentation
  workers were not deterministic. The replacement v2 baseline passes its full replay gate. The
  publication hold now comes from the generated and simulated arms, which still require an
  audited, registration-qualified paired source manifest and frozen generator checkpoints.
- **GO for the Phase 1 three-arm screen only after:** a sequence-disjoint, training-authorised
  paired source is frozen and its RGB/thermal registration passes the threshold, and DiffV2IR and
  PID checkpoints pass a fixed inference fixture. The deterministic detector-baseline replay
  condition is now met. The FLIR test-pair manifest, v1 detector runs and timing pilot remain
  supporting evidence; none of them waive the paired-source requirements.
- **NO-GO for the full grid until:** pilot variance determines the number of confirmatory seeds
  and the RFS decision rule and grouped cross-validation protocol are frozen. Grid expansion now
  de-duplicates identical zero-generated controls across generator, budget and sensor factors.

This gate distinguishes a runnable scaffold from a result that can support a scientific claim.

---

### Phase 0 — scaffold and instrumentation

**CPU and CUDA pilots passed.** The core is present and covered by tests. Phase 0 is an
engineering and metric sanity gate; it does not establish the research hypothesis.
Class-conditioned RFS aggregation and a frozen undefined-statistic policy remain Phase 1 gates.

| Component | File | Status |
|---|---|---|
| Differentiable IR sensor chain (MTF, NETD, FPN/NUC, AGC + 8-bit) | `src/aero_ir/sensor/` | implemented, differentiability tested |
| Radiometric Fidelity Score R1–R8 | `src/aero_ir/rfs/stats.py` | pooled diagnostic implemented; class conditioning pending |
| Distributional distances + real-set sampling floor | `src/aero_ir/rfs/distances.py`, `report.py` | implemented |
| Mixing budget semantics (`fixed_total` vs `additive`) | `src/aero_ir/data/mixing.py` | implemented, tested |
| Experiment grid E1–E5 as configuration | `configs/experiment/` | declared |
| Protocol, RFS spec, roadmap | `docs/` | written |
| FLIR audit, immutable detector manifest and loader | `src/aero_ir/data/flir.py`, `data/registry.py` | implemented; local release passed |
| Train-only analytics16 detector preprocessing | `src/aero_ir/data/preprocess.py` | implemented; 6076-8097 DN window frozen |
| Pinned YOLOX FLIR adapter, evaluator and run lifecycle | `src/aero_ir/detect/` | v1 replay exceeded tolerance; deterministic fix passes an exact twin GPU smoke; clean full v2 baseline and metric replay passed |
| Anti-UAV300 archive/video/annotation audit | `src/aero_ir/data/antiuav.py`, `scripts/audit_antiuav300.py` | implemented; archive, extraction and timing passed; annotation/registration holds recorded |
| Anti-UAV300 train-only registration audit | `src/aero_ir/data/antiuav_registration.py`, `scripts/audit_antiuav300_registration.py` | implemented; global target-box calibration failed train and held-out validation gates |
| Anti-UAV300 image-conditioned dense registration | `src/aero_ir/registration/superfusion.py`, `src/aero_ir/registration/protocol_v2.py`, `scripts/train_antiuav300_registration_v2.py`, `scripts/audit_antiuav300_dense_registration.py` | v1 and distinct 300-epoch v2 fits completed; v2 frozen screen failed (29.30%/22.57% joint versus 95%); exhaustive audit and generator training HOLD |
| Anti-UAV300 native-IR YOLOX adapter | `src/aero_ir/data/antiuav300_ir.py`, `src/aero_ir/detect/yolox_antiuav300_exp.py` | smoke and 20-epoch diagnostic passed; corrected 300-epoch standard schedule CPU-verified, GPU run pending |
| Anti-UAV410 external detection adapter | `src/aero_ir/data/antiuav410.py`, `scripts/audit_antiuav410.py`, `scripts/prepare_antiuav410.py` | implemented; local archive/test manifest passed; 64 invalid test positives excluded and counted |

**Exit criterion met.** `make smoke` passes, and the Phase 0 pilot passes on the intended RTX
4090 compute device. The smoke test needs neither GPU nor data; the pilot times a sensor-chain
forward/backward pass and checks that RFS separates faithful, degraded and polarity-inverted
fixtures. The recorded CUDA result has finite gradients and includes timing and peak-memory data.

---

### Phase 1 — data layer and the E1 protocol transfer

**The gate.** E1 transfers the fixed-total mixing design of `vanherle2022` to infrared detection;
it is **not** an exact reproduction. The source study used DIMO, Mask R-CNN/ResNet-101 and its own
training recipe, whereas E1 uses FLIR and YOLOX-s. An exact reproduction must use the authors'
DIMO code and data and be reported separately. E1's purpose is to qualify this repository's IR
baseline and establish the three-arm curve under a fully recorded protocol.

**Implement**

| File | What |
|---|---|
| `src/aero_ir/data/flir.py` | COCO/T-linear audit, content-addressed detector manifest and lazy analytics16 loader are implemented |
| `src/aero_ir/data/flir_pairs.py` | immutable official `video_test` pair manifest and conservative cross-modal box audit are implemented; training and transform calibration are explicitly prohibited because no official train/validation pairing exists and registration did not qualify |
| `src/aero_ir/data/registry.py` | the FLIR loader is connected to the configured lazy dataset interface; images are `(H, W)` uint16 DN arrays and boxes use COCO `(x, y, w, h)` |
| `src/aero_ir/detect/yolox_adapter.py` | the pinned backend and manifest-aware upstream experiment pass CPU and both GPU engineering checks; effective-batch accumulation is implemented and tested |
| `src/aero_ir/detect/evaluate.py`, `yolox_evaluator.py` | complete COCO AP/AR and per-class metrics are implemented, fixture-tested and wired to formal training |
| `src/aero_ir/detect/yolox_run.py`, `scripts/run_flir_yolox.py` | content-addressed prepare/execute/finalise lifecycle, isolated replay and bounded synchronized timing are implemented and timing-tested; protocol review requires clean Git |
| `src/aero_ir/cli.py` | dispatch for `run` |
| `scripts/run_grid.py` | replace the `pass` with a launcher call |

**Run**

```bash
export AERO_DATA_ROOT=/path/to/extracted
export AERO_FLIR_ROOT="$AERO_DATA_ROOT/FLIR_ADAS_v2"
# optional but required for a publication manifest:
export AERO_FLIR_ARCHIVE=/path/to/downloaded-release.zip
bash scripts/download_flir.sh          # verifies the official directory layout
make audit-flir                        # writes experiments/flir_data_audit.json
make manifest-flir                     # writes experiments/flir_trainval_manifest.json
make manifest-flir-pairs               # freezes official video_test pairs; CPU, evaluation only
make preprocess-flir                   # writes experiments/flir_preprocess.json
make prepare-flir-yolox                # writes filtered COCO views; uses no GPU
make pilot-flir-rfs                    # writes experiments/flir_rfs_pilot.json
python scripts/run_grid.py e1_reproduce --dry-run
# Two independent 20-epoch subset runs plus exact determinism comparison:
bash scripts/run_flir_determinism_smoke.sh
# The two engineering-subset GPU smokes have passed. Preserve their records:
make record-flir-yolox-smoke \
  RUN_DIR=experiments/yolox_runs/flir_real_only_smoke_seed0
# The v1 baseline artifacts remain valid and can be verified without using a GPU:
.venv/bin/python scripts/verify_run.py \
  --run experiments/yolox_runs/flir_real_only_full_seed0_v1/run_manifest.json
# Its full replay completed but failed the deterministic tolerance. Do not rerun or alter v1.
# The twin deterministic GPU smoke and clean 300-epoch v2 baseline have completed.
# The launcher refuses to overwrite that baseline; retain this as its reproduction command:
bash scripts/run_flir_v2_baseline.sh
# The frozen v2 manifest replay passed; this is its reproduction command:
bash scripts/replay_flir_v2_baseline.sh
# after every HOLD gate above is cleared:
make e1
```

**Exit criterion.** First, a real-only YOLOX-s arm must complete, produce plausible COCO metrics,
and replay from its manifest. Then run the three-arm screening grid with matched seeds. Record the
curve as an IR protocol-transfer result; do not require it to copy the shape of a different model
and domain. Three seeds are for screening only and do not support a confidence-interval claim.

**Compute.** The Cartesian grid has 42 cells; collapsing the duplicated real-only generator
controls leaves **39 runs**, YOLOX-s from scratch
(300 epochs — the expensive phase, by construction, since scratch training is the condition
being tested).

**Also delivers a result on its own.** `synthetic_baseline` versus `diffv2ir` at matched
ratio and budget answers a question the field mostly assumes: *does the generative step beat
a crude simulator?* If it does not, that is a finding, and it reframes everything after it.

---

### Phase 2 — the controlled experiment (F1, F3)

**Implement**

| File | What |
|---|---|
| `src/aero_ir/utils/manifest.py` | wire `RunManifest` into every run; dataset checksums |
| `scripts/verify_run.py` | re-execute from a manifest, diff metrics within tolerance |
| W&B logging | `tracking.wandb` in `configs/config.yaml` |

**Run**

```bash
make e2     # train: [scratch, pretrained] x gen_ratio x seed   -> the sign-flip test
make e3     # budget_mode: [fixed_total, additive] x gen_ratio x seed
```

**Exit criterion.** Screen both questions with three matched seeds, then power and run the
preregistered confirmatory contrasts before attaching a bootstrap confidence interval:

1. **Does the sign of `dAP` change with initialisation alone?** E2 locks epochs, batch semantics,
   optimiser, learning rate, warm-up and augmentation across the two arms; only initial weights
   differ. A separate recipe ablation is required before attributing any effect to training length.
2. **Does `fixed_total` differ from `additive`?** These answer different questions —
   *substituting* generated for real data versus *adding* it. Reporting them separately is
   what separates "generated data harms" from "having less real data harms".

A null result here is still a result. Its scope is this detector, generator and domain; it does
not establish field-wide robustness without held-out-generator and held-out-domain confirmation.

**Compute.** E2: 2 inits x 4 ratios x 3 screening seeds = 24 runs. E3's Cartesian grid has
72 cells. Additive mixing excludes ratio 1.0 because a 100% generated fraction is undefined when
the real count is held constant. Collapsing nine duplicate zero-generated E3 cells leaves
**87 screening runs** across E2 and E3, followed only by powered confirmation of preregistered
contrasts.

---

### Phase 3 — label audit and failure-mode decomposition (F4, F5)

No new training. This phase re-analyses the runs Phase 2 produced, and it is where the
analysis stops being a single number.

**Implement**

| File | What |
|---|---|
| `src/aero_ir/data/labels.py` | `audit()` — IoU drift, centroid shift, area ratio, contrast floor. Reuses `rfs.stats.target_snr` so the audit and the fidelity diagnostic share one definition of contrast |
| `src/aero_ir/analysis/failure_modes.py` | `stratified_ap()` over target pixel area, contrast quartile, polarity, clutter |

**Exit criterion.** Every reported result carries its label-exclusion rate, and `dAP` is
decomposed by stratum. Concretely: the answer to *"where did the effect actually land?"* is a
table, not a guess. If the effect is confined to the tiny-target bucket, that is the finding
and it reshapes the paper.

**Compute.** None beyond inference over existing checkpoints.

---

### Phase 4 — RFS predictive power (N1)

**The core contribution.** Everything before this establishes that the effect is real and
measurable; this phase asks whether it can be *predicted* — and therefore controlled.

**Implement**

| File | What |
|---|---|
| `src/aero_ir/curate/selectors.py` | `RFSSelector` (per-sample RFS, `closest` and `coverage` modes), `PerceptualSelector` (per-sample distance to the real feature centroid), `MarginalAPSelector` (gradient-alignment proxy) |
| `src/aero_ir/analysis/predictive_power.py` | compare FID / LPIPS / SSIM / SDQM / CCDM / RFS scalar / RFS vector; Spearman and cross-validated R², with folds grouped by generated pool and leave-one-generator/domain-out tests |

**Run**

```bash
make e4     # curation: [none, random, fid_topk, rfs_topk, marginal_ap] x ratio x seed
```

**Exit criterion.** A single table: predictive power of each fidelity measure for `dAP`.

- **H1** holds only if a preregistered interval shows FID and LPIPS below a stated predictive
  threshold on held-out groups.
- **H2** holds only if RFS improves held-out predictive performance over both perceptual and
  recent detection-specific metrics, with uncertainty reported.
- **H4** holds if `rfs_topk` beats `random` and `fid_topk` at an equal budget.

Randomly splitting run rows is prohibited: seeds sharing the same generated pool are not
independent data points. If RFS loses, say so plainly. Results confined to FLIR or one generator
must be described as domain-specific, not as a general training-data metric.

**Compute.** 5 policies x 2 ratios x 3 seeds = **30 runs**, pretrained.

---

### Phase 5 — small-target external validation and sensor matching

Anti-UAV410 is an IR-only tracking benchmark, not a paired RGB/IR detection training set. It may
be adapted to frame-level detection only with sequence-disjoint sampling and explicit visibility
handling; it cannot directly provide RGB inputs for a visible-to-IR generator. Use Anti-UAV300's
paired training sequences for generator development and Anti-UAV410 only as external thermal
evaluation. No validation/test frame, label or paired visible image may enter generation or
training. A genuine non-acquirable-scenario coverage claim is deferred to Phase 8 unless an
independent source of those scenarios is specified.

The local Anti-UAV300 release is currently usable for a filtered IR-only training pilot, not yet
for paired generator training. Its official split manifests are sequence-disjoint and temporally
paired, but zero-area present boxes must be excluded. A train-fitted global RGB-to-IR box
transform fails both its training sanity and held-out validation gates. The replacement
image-conditioned dense matcher completed its 300-epoch train-only fit, but its unchanged
parameters failed the sequence-balanced screen (74.22% train, 67.54% validation joint pass
against 95%). The exhaustive held-out validation audit was correctly not run. The
display-referred 8-bit MP4 stream is also not a substitute for radiometric DN data. The local
Anti-UAV410 external-test adapter is ready, but that does not resolve the paired-source hold.

**Implement**

| File | What |
|---|---|
| `src/aero_ir/data/antiuav.py`, `antiuav300_ir.py` | archive/split audit plus manifest-locked native-IR smoke preparation; label-independent sampling, invalid-positive exclusion and negative retention are implemented |
| `src/aero_ir/data/antiuav_registration.py` | sequence-balanced robust target-box transform fit on train only and applied unchanged to validation/test; qualification fails |
| `src/aero_ir/registration/superfusion.py`, `scripts/audit_antiuav300_dense_registration.py` | pinned public dense matcher, sub-pixel box-flow inversion and fixed dense/geometry gates; completed 300-epoch fit failed sequence-balanced GPU qualification |
| `src/aero_ir/detect/antiuav300_yolox.py`, `yolox_antiuav300_exp.py` | one-class prepared-PNG loader and deterministic YOLOX-s smoke/standard profiles; no paired RGB input is used |
| `src/aero_ir/data/antiuav410.py` | audited, content-addressed test-only tracking-to-detection manifest and COCO export; invalid positives are excluded and negatives retained |
| `src/aero_ir/data/registry.py` | lazy Anti-UAV410 external-test loader preserving `sequence_id`, visibility, attributes and `target_pixel_area_bin`; Anti-UAV300 training remains registration-gated |
| `src/aero_ir/sensor/fit.py` | `fit_sensor_params()` — NETD from the noise PSD floor (R5), MTF cutoff from the spectrum roll-off (R4), column FPN from the variance ratio (R8), AGC clip points from the histogram (R7) |

**Run**

```bash
export AERO_ANTIUAV300_ROOT=/path/to/Anti-UAV300
export AERO_ANTIUAV300_ARCHIVE="$AERO_ANTIUAV300_ROOT/Anti-UAV300.zip"  # recommended
bash scripts/download_antiuav.sh
make audit-antiuav300       # writes experiments/antiuav300_data_audit.json
make audit-antiuav300-registration  # CPU; train fit, validation gate, report-only test
bash scripts/run_antiuav300_registration.sh  # GPU; screen, then exhaustive audit only on pass
bash scripts/run_antiuav300_registration_v2.sh  # GPU; frozen v2, then unchanged screen/full test
bash scripts/run_antiuav300_registration_v4.sh audit  # GPU; corrected audit, NOT training
bash scripts/run_antiuav300_registration_v4_smoke.sh  # GPU; 3 FP32 optimizer steps, no weight saves
# v1--v3 runners are historical only; their gates do not qualify paired generator training.
make pilot-antiuav300-rfs   # train-only; writes experiments/antiuav300_rfs_pilot.json
make prepare-antiuav300-ir-yolox  # CPU; uniform native-IR subset + full hash preflight
AERO_ANTIUAV300_RUN_ID=antiuav300_native_ir_standard_seed0_e300_v1 \
  AERO_ANTIUAV300_EPOCHS=300 \
  AERO_ANTIUAV300_TRAINING_PROFILE=standard \
  AERO_ANTIUAV300_EVAL_INTERVAL=10 \
  bash scripts/run_antiuav300_ir.sh  # GPU; requires a clean Git checkout

export AERO_ANTIUAV410_ROOT=/path/to/Anti-UAV410
export AERO_ANTIUAV410_ARCHIVE="$AERO_ANTIUAV410_ROOT/Anti-UAV410.zip"
make audit-antiuav410       # CPU; CRC plus every split/label/frame alignment
make prepare-antiuav410     # CPU; immutable test manifest plus COCO export

# Do not run make e5 yet: Anti-UAV300 failed paired-source registration and generator gates.
```

Re-running `scripts/run_antiuav300_registration.sh` with the existing screen verifies its content
hash, dataset root, checkpoint hash and frozen gate computation, then reports **HOLD** with exit
code 3 without requiring CUDA or overwriting the result. A new training method needs a distinct,
pre-registered run; moving or deleting this screen does not make the failed gate pass.

**Exit criterion.** On sequence-disjoint external data, is `dAP > 0`, and does the fitted sensor
model beat both the generic profile and no sensor model? This tests transfer beyond the urban
domain; it does not by itself prove benefit in unobserved operational scenarios.

**Compute.** The Cartesian grid has 36 cells; collapsing six duplicate zero-generated sensor
controls leaves **30 runs**.

---

### Phase 6 — sensor-in-the-loop generation (N2)

Everything up to here applies the sensor model as post-processing. This phase puts it *inside*
the loop: a detection loss back-propagated through the sensor chain into the generator, so
generation is optimised for the detector's decision statistics rather than for perceptual
realism. The sensor stages are already `torch.nn.Module`s with a straight-through quantiser
for exactly this reason, and `tests/test_sensor.py::test_chain_is_differentiable_end_to_end`
guards the gradient path.

**Implement**: `src/aero_ir/generate/tev.py`, generator fine-tuning with the sensor chain and
a frozen detector in the graph.

**Open questions to resolve here, not before**: which sensor parameters are identifiable from
imagery alone, and whether back-propagating through AGC needs the soft surrogate
(`AGCQuantise(soft_quantise=True)`) rather than the straight-through estimator.

**Exit criterion.** Generated data from the in-loop generator beats post-processed generated
data on `dAP` at an equal budget.

**Compute.** Generator fine-tuning dominates. Budget ~20 detector runs plus generator training.

---

### Phase 7 — deployment track

Small in effort, and it is the difference between a research repository and one that reads as
deployable. Can be done any time after Phase 2 has a trained checkpoint.

**Implement**: `src/aero_ir/deploy/export.py` — ONNX export, INT8 post-training quantisation
calibrated on the real training split, latency percentiles on a fixed device profile.

```bash
make deploy-bench
```

**Exit criterion.** One latency-versus-mAP curve. The curve is the deliverable, not a single
quantised number: what matters operationally is *where* accuracy starts to fall off.

**Compute.** Hours on one GPU.

---

### Phase 8 — radiometrically consistent 3D generation (N3)

The only track that addresses the original constraint — imagery that *cannot be acquired* —
rather than restyling imagery that already exists. Scaffolded in `src/aero_ir/scene3d/`;
the plan is in `src/aero_ir/scene3d/plan.md`.

Physically consistent thermal 3D reconstruction has been published and is evaluated on
reconstruction quality. The open question is whether such views work as **training data**,
which is what Phases 1–5 build the instrument to measure. The differentiation is the
evaluation axis and the small-target regime, not the renderer.

**Do not start this before Phase 5 reports.** Without a working `dAP` measurement in the
held-out-scenario setting, there is nothing to evaluate the rendered views against, and the
phase collapses into a reconstruction-quality exercise that has already been done.

---

### Optional — E6 capacity

Does the effect of generated data depend on detector capacity? Every published result on this
question is single-model, so nobody knows. `make e6` sweeps YOLOX tiny/s/m/l. Run it after E4
reports; if the sign of `dAP` is capacity-dependent, that constrains how any of these
conclusions may be stated.

**Compute.** 4 sizes x 3 ratios x 3 seeds = 36 runs.

---

### Phase 9 — outputs

- `scripts/make_report.py` regenerates all five figures and three tables from the runs table.
  Figures are derived, never hand-edited.
- Preprint; model card, data card, environment lock.
- Every claim traceable to a run id in `experiments/`.

---

## Moving this repository to a GPU machine

Nothing here is bound to the machine it was written on. `data/` and `experiments/` are
git-ignored, so a clone carries code and configuration only.

### 1. Transfer

```bash
# preferred - carries history
git clone <remote> aero-ir && cd aero-ir

# or, direct copy; exclude generated state explicitly
rsync -av --exclude '.git' --exclude 'data' --exclude 'experiments' \
      --exclude '__pycache__' --exclude '.pytest_cache' \
      aero-ir/ user@gpu-host:~/aero-ir/
```

### 2. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# install torch + torchvision first using the command generated for the host by
# https://pytorch.org/get-started/locally/ ; do not copy a stale CUDA wheel URL

pip install -e ".[torch,track,detect,registration,deploy,dev]"
pre-commit install
```

`torch` is an optional extra rather than a hard dependency precisely so the analysis and RFS
code stay installable on a machine without CUDA.

For the RTX 4090 check on this checkout, an isolated environment was created as follows (the
directory is ignored by Git):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e ".[dev,detect,registration]"
make smoke PY=.venv/bin/python
```

The environment passes the full CPU smoke suite and the CUDA Phase 0 pilot on a healthy RTX 4090.
An initially tested GPU host returned error 999 from both the vendor `deviceQuery` binary and
PyTorch, even with CUDA library overrides removed; moving the same environment to a healthy GPU
cleared the error. That failure was host-level rather than a repository or virtual-environment
failure.

### 3. Verify the transfer — before touching data or GPUs

```bash
make smoke
```

Runs lint, formatting, the full available test suite, and an RFS end-to-end check on synthetic
scenes. Sensor tests require PyTorch but not a GPU. On a correct
transfer a radiometrically degraded set scores far from the real set and the report names
the components that failed. If this passes, the code arrived intact.

### 4. Data

```bash
export AERO_DATA_ROOT=/mnt/data/flir-release/extracted
export AERO_FLIR_ROOT="$AERO_DATA_ROOT/FLIR_ADAS_v2"
bash scripts/download_flir.sh               # verifies the official directory layout
make audit-flir
make pilot-flir-rfs

export AERO_ANTIUAV300_ROOT=/mnt/data/Anti-UAV300
export AERO_ANTIUAV300_ARCHIVE="$AERO_ANTIUAV300_ROOT/Anti-UAV300.zip"
bash scripts/download_antiuav.sh
make audit-antiuav300
make audit-antiuav300-registration
bash scripts/run_antiuav300_registration.sh  # GPU; does not proceed after a failed screen
make pilot-antiuav300-rfs
make prepare-antiuav300-ir-yolox
AERO_ANTIUAV300_RUN_ID=antiuav300_native_ir_standard_seed0_e300_v1 \
  AERO_ANTIUAV300_EPOCHS=300 \
  AERO_ANTIUAV300_TRAINING_PROFILE=standard \
  AERO_ANTIUAV300_EVAL_INTERVAL=10 \
  bash scripts/run_antiuav300_ir.sh  # native-IR subset only; does not run E5

export AERO_ANTIUAV410_ROOT=/mnt/data/Anti-UAV410
export AERO_ANTIUAV410_ARCHIVE="$AERO_ANTIUAV410_ROOT/Anti-UAV410.zip"
make audit-antiuav410
make prepare-antiuav410
```

Datasets are gated and are never fetched automatically. The scripts verify layout and record
checksums after manual placement.

### 5. Execution model

The available target is one RTX 4090 with 24 GB. The official YOLOX recipe recommends a global
batch of 64 across eight GPUs (eight images per GPU); do not assume batch 64 fits on one card.
The adapter exposes per-device batch size and gradient accumulation explicitly and preserves the
effective batch and learning-rate semantics across arms. The first end-to-end detector smoke used
batch 8 and FP16 before accumulation was enabled. It completed the 128-image training subset and
64-image validation subset in 122.981 seconds, logged 1,812 MiB CUDA allocation, and wrote valid
checkpoints. The later batch-8 x 8-step accumulation smoke completed in 46.909 seconds with two
optimiser steps. Neither tiny-subset wall time nor YOLOX's asynchronous iteration log is a valid
full-data runtime estimate. The bounded timing runner therefore uses the full training split and
normal mosaic/mixup/multiscale state, synchronises CUDA, excludes warm-up and records throughput,
loss finiteness, input sizes, peak memory, software and hardware in a replayable manifest. Its
batch-32 x 2-step, eight-worker run completed 160 microbatches across six sizes at 19.21 images/s
including cold-size startup, with 9,827 MiB peak allocation. Novel sizes incurred roughly
43-second one-time CUDA/cuDNN initialization stalls; a repeated 672-pixel block averaged about
0.20 seconds per microbatch. Batch 32 therefore has ample memory headroom and is the frozen
single-GPU microbatch policy; raw short-pilot throughput remains a conservative runtime estimate.
The corresponding 300-epoch real-only run completed in 5.987 hours and peaked at 11,104 MiB,
substantially improving the conservative 47-hour pilot projection. Its final mAP@0.5:0.95 is
0.3513. The normally zero L1 term is expected: YOLOX enables its optional L1 box-regression loss
only after mosaic is disabled for the final no-augmentation phase; this run activated it at
displayed epoch 285, after which logged L1 losses were approximately 0.3--0.5.
Its full replay completed in 5.736 hours and reached 0.3574 mAP@0.5:0.95, 0.00613 above the
original and outside the frozen 0.002 tolerance. The schedule matched, but augmentation-worker
RNGs did not: upstream YOLOX 0.3.0 seeds them from UUIDs. The current adapter replaces that path
with deterministic worker seeds and strict CUDA algorithm settings. Its twin GPU smoke produced
exactly matching checkpoints, predictions, scientific metrics and normalised training traces. A
new clean full v2 baseline now reaches 0.35565 mAP@0.5:0.95 and passes static manifest
verification. Its full 300-epoch replay also passes the complete nested-metric comparison at the
frozen 0.002 tolerance, so the real-only detector baseline is now reportable under its recorded
manifest and source tag.

On this checkout on 2026-09-05, the Phase 0 CUDA fixture passed on an RTX 4090 with
`torch==2.9.0+cu128`: 1.158 ms mean forward/backward time over 10 measured iterations and
43.4 MiB peak allocated memory for an `[8, 1, 256, 320]` input. These values qualify the sensor
fixture only. The full-data timing result above, rather than the sensor fixture, governs detector
runtime and memory planning.

```bash
# inspect the grid; this does not launch jobs
python scripts/run_grid.py e3_mixing_ratio --dry-run

# full-data timing passed; grid launch remains disabled until a clean-snapshot baseline manifest
# has completed and replayed successfully
```

Set `tracking.wandb.mode=offline` on a host without outbound network and sync later with
`wandb sync`.

---

## Compute budget

Only the Phase 1 row now has a measured full-run RTX 4090 wall time. Other rows remain
placeholders until their own pilots. Before approving the grid, also benchmark generator
inference and record checkpoint size and total wall time. Counts below are raw Cartesian cells:
identical zero-generated controls must be scheduled once and referenced by the other cells rather
than retrained.

| Phase | Screening runs | Init | Epochs | RTX 4090 runtime |
|---|---|---|---|---|
| 1 (E1) | 42 | scratch | 300 | 5.987 h measured real-only baseline |
| 2 (E2) | 24 | mixed, schedule matched | 100 | TBD by pilot |
| 2 (E3) | 72 | pretrained | 100 | TBD by pilot |
| 4 (E4) | 30 | pretrained | 100 | TBD by pilot |
| 5 (E5) | 36 | pretrained | 100 | TBD by pilot |
| **screening total** | **204** | | | **TBD** |
| E6 (optional) | 36 | pretrained | 100 | TBD by pilot |

Confirmatory reruns selected by power analysis are additional. Phases 3, 7 and 9 add no detector
training. Phase 6 is dominated by generator fine-tuning and is budgeted separately.

**Cut if the budget is tight**, in this order: E6 entirely; then E3's `fixed_total` half
(keep `additive`, which is the operationally meaningful mode); then E1's ratios 0.6 and 0.8.
Never cut seeds — a two-seed result cannot support a claim about a few AP points.

Storage: budget roughly 200 GB for datasets, generated sets and checkpoints combined.

---

## The pipeline

```
S1  Scenario definition          acquirable vs non-acquirable scenarios; coverage gap made explicit
S2  Physics-conditioned          T / epsilon / V decomposition + differentiable IR sensor model
    generation                   (NETD, MTF, AGC + 8-bit quantisation, FPN / NUC residual)
S3  Radiometric Fidelity Score   diagnostic vector, not a scalar:
    (RFS)                        target-background dT, thermal polarity, radial spectrum,
                                 noise PSD, per-class target SNR, target pixel-area
S4  Utility-aware curation       select / weight generated samples under a fixed budget
S5  Controlled training          (pretrained on/off) x (fixed-N / additive) x (mix ratio)
                                 x (domain) x screening seeds + powered confirmation
S6  Diagnostics + deployment     failure-mode decomposition; ONNX -> TensorRT INT8;
                                 latency vs mAP trade-off
```

Full design rationale: [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## What the controlled protocol fixes

Prior synthetic-data studies vary across five design properties that can change the result.
Each one is measured or controlled here rather than treated as an assumption.

| | Common property | Why it invalidates the conclusion | Handled by | Phase |
|---|---|---|---|---|
| F1 | Detector initialisation differs across studies | Initialisation can interact with distribution shift and training dynamics | `train.pretrained` as an explicit factor while the schedule is fixed | 2 |
| F2 | Generated data derived from the *same scenes* as the real data | Adds noise without adding coverage — the entire point of synthetic data is untested | `data.coverage_split` | 5 |
| F3 | Mixing ratio confounded with total dataset size | A ratio increase may mean less real data *or* more total data; the two imply opposite conclusions | `mixing.budget_mode` | 2 |
| F4 | Labels transferred across the generation step, unaudited | If generation deforms object boundaries, label noise enters silently | `data.labels.audit` | 3 |
| F5 | A single aggregate AP number | Cannot tell *which* failure mode moved, so the analysis terminates in a question mark | `analysis.failure_modes` | 3 |

## Repository layout

```
docs/                     problem statement, experiment protocol, RFS spec, datasets, roadmap
configs/                  Hydra tree - every experiment declared here, none in scripts
  experiment/             E1-E5, each with its sweep and success criterion
src/aero_ir/
  sensor/                 differentiable IR sensor chain          [Phase 0 pilot ready]
  rfs/                    radiometric diagnostic R1-R8            [class conditioning pending]
  data/                   loaders, mixing budget, label audit     [mixing done]
  generate/               generator adapters behind one Protocol
  curate/                 selection policies
  detect/                 detector adapter, COCO metrics, dAP     [dAP done]
  analysis/               failure modes, predictive power
  deploy/                 ONNX / INT8 / latency
  scene3d/                N3 scaffold + plan.md
scripts/                  dataset access, grid expansion, report, run verification
tests/                    79 tests; no GPU or dataset required
```

## Mapping to industry requirements

Public job descriptions in defence EO/IR R&D repeatedly ask for the same capabilities.
This table exists so a reviewer can find the corresponding code in one step.

| Required capability | Module | Entry point |
|---|---|---|
| Generation of synthetic EO/IR imagery for training | `aero_ir.generate` | `configs/generator/` |
| Construction and curation of a training database | `aero_ir.data`, `aero_ir.curate` | `configs/curation/` |
| Automatic target detection / recognition models | `aero_ir.detect` | `configs/detector/` |
| Physics-based sensor and scene modelling (M and S) | `aero_ir.sensor`, `aero_ir.scene3d` | `configs/sensor/` |
| On-device optimisation and profiling | `aero_ir.deploy` | `make deploy-bench` |
| Reproducible experiment control and configuration management | Hydra + DVC + W and B | `configs/`, `dvc.yaml` |

## Data

Public datasets only. No proprietary imagery, labels, or specifications are used or referenced.

| Dataset | Role | Phase |
|---|---|---|
| Teledyne FLIR ADAS Thermal v2 | IR protocol-transfer anchor; pair manifest required | 1 |
| LLVIP | aligned low-light visible-IR pairs | optional |
| Anti-UAV300 / Anti-UAV410 | paired development (registration HOLD) / audited IR-only external evaluation | 5 |
| DroneVehicle | aerial viewpoint, oriented boxes | optional |

See [`docs/datasets.md`](docs/datasets.md) for licences and access.

## Provenance

This repository is built from public sources only, and is written so that fact is checkable
rather than merely asserted.

- **Prior results** are cited in [`docs/references.md`](docs/references.md). A statement about
  what previous work found is not made in this repository without a reference beside it.
- **Data** is public and gated only by the providers' own request forms. See
  [`docs/datasets.md`](docs/datasets.md).
- **Baselines** — YOLOX, DiffV2IR and PID have public code/checkpoint routes. Exact revisions and
  artifact hashes must be pinned in every run. E1 starts from the documented YOLOX 300-epoch
  scratch recipe but is an IR protocol transfer, not a reproduction of the DIMO experiment.
- **No proprietary material** of any kind: no imagery, labels, sensor specifications,
  requirement documents, internal results, or organisation names. This is enforced by
  `.gitignore` and stated as the first rule in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Reproducibility

The FLIR YOLOX runner implements the reproducibility contract: preparation freezes the config,
git SHA and dirty state, full-split and preprocessing checksums, code hashes, seed and execution
environment; finalisation adds hardware, runtime, predictions and the complete metric set. Replay
uses a temporary output root so it cannot overwrite the recorded run. Generated-data runs must
also bind their pair manifest and generator checkpoint before they are reportable. `make verify
RUN=<manifest.json>` checks artifact integrity, and `make replay RUN=<manifest.json>` re-executes a
trusted local manifest and compares its deterministic metrics within the declared tolerance.
When the checkout has advanced, read-only verification may validate changed tracked inputs from
the manifest's recorded Git commit and lists them under `verified_from_git`. Execution is refused
in that state; a replay may never silently substitute current source for the recorded source.

The claims concern effect signs of a few AP points. Use matched seeds and bootstrap the paired
per-seed differences. Three seeds are screening evidence only. Run a pilot to estimate variance,
perform power analysis for the smallest effect of interest, and use at least ten matched seeds
for any percentile-bootstrap headline contrast; recent methodological work shows small-sample
bootstrap tests can substantially understate false positives. Predictive-power cross-validation
must be grouped by generated pool and include held-out-generator/domain evaluation.

## Licence

MIT. See [`LICENSE`](LICENSE).
