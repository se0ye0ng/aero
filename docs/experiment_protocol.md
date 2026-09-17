# Controlled experiment protocol

Every factor below is an explicit dimension of the run grid. Nothing is fixed by convention.

## Factors

| Factor | Levels | Rationale |
|---|---|---|
| `train.pretrained` | `false`, `true` (COCO) | **F1.** Initialization ablation with optimizer, LR, warm-up, augmentation, effective batch and epoch budget held fixed |
| `mixing.budget_mode` | `fixed_total`, `additive` | **F3.** Under `fixed_total`, raising the generated ratio removes real data. Under `additive`, real data is held constant and generated data is added. The two answer different questions and imply opposite conclusions |
| `mixing.gen_ratio` | fixed-total: 0–1; additive: 0–0.8 | 1.0 is undefined for additive mixing because the real count remains nonzero |
| `data.coverage_split` | `same_scene`, `held_out_scenario` | **F2.** `same_scene` adds no scenario coverage. `held_out_scenario` withholds a slice from real training and is valid only when generation has an independent, non-evaluation source for that slice |
| `curation.policy` | `none`, `random`, `fid_topk`, `rfs_topk`, `marginal_ap` | H4. Compared at an equal generated-sample budget |
| `sensor.model` | `none`, `eo_ir_default`, `matched` | N2. `matched` fits the sensor model to the real set before applying it to generated data |
| `data.domain` | `flir_urban`, `antiuav_small` | generality across regimes. The small, low-contrast regime is where operational IR detection actually lives |
| `detector.arch` | `yolox_s`, `yolox_tiny`, one modern detector | `yolox_*` preserves comparability with the reported result. The third is a generality check only |
| `seed` | 3 matched screening seeds; powered confirmation | bootstrap headlines require pilot power analysis and at least 10 matched seeds |

## Experiments

### E1 — IR protocol transfer
Fix `pretrained=false`, `coverage_split=same_scene`, `curation=none`, `sensor=none`,
`domain=flir_urban`. Sweep `gen_ratio`. The fixed-total design is transferred from
`vanherle2022`, but the different domain/model means this is not an exact reproduction.
**Success = a qualified, replayable real-only baseline followed by a complete three-arm curve.**

### E2 — pretraining ablation (the sign-flip test)
E1 grid with `pretrained` crossed in. Primary readout: does the sign of
`dAP = AP(real + gen) - AP(real)` change with initialisation alone?

### E3 — ratio x budget
Cross `gen_ratio` with `budget_mode`. Separates "generated data is harmful" from
"replacing real data with generated data is harmful". These are not the same claim.

### E4 — curation
Fixed generated-sample budget, varying `curation.policy`. Tests H4 and, by comparing
`fid_topk` against `rfs_topk`, tests H1 and H2 directly.

### E5 — sequence-disjoint small-target external validation
Use paired Anti-UAV300 training sequences for visible-to-IR development and IR-only Anti-UAV410
as external evaluation through the manifest-locked tracking-to-frame-detection adapter. Never expose
validation/test frames or annotations to the generator. Do not claim a non-acquirable-scenario
coverage result without an independent source for those scenarios.

The audited Anti-UAV300 global RGB-to-IR target-box transform fails its frozen train and held-out
validation criteria. Anti-UAV300 may therefore support a native-IR detector pilot after invalid
annotations are filtered, but it is not an aligned paired source for generator training.
An image-conditioned SuperFusion DenseMatcher qualification is implemented as the corrective
path. Its public checkpoint initializes a fixed 300-epoch fit using only a sequence-balanced
cache from the official training split. It must then pass the original 95% box-transfer
threshold plus fixed flow-validity and edge-alignment criteria on held-out validation before
paired generator training is permitted.
The native-IR engineering adapter implements that pilot with label-independent uniform sampling,
retained negative frames and recorded invalid-positive exclusions. Its output must not be used as
evidence that paired generator development or E5 is unblocked.

#### Anti-UAV300 registration protocol v2 (frozen 2026-09-16 before execution)

Protocol v1 completed 300 epochs but failed its unchanged sequence-balanced screen (74.22% train,
67.54% validation joint box pass versus 95%). Its checkpoint, report and thresholds remain
immutable. Protocol v2 is a distinct train-only intervention, not a replay or threshold change:

- initialize from the train-only v1 checkpoint `a4c8aafe...2388a`;
- cache all 141,816 usable official training pairs, without opening validation or test;
- draw 16 pairs per training sequence per epoch from deterministic rotating permutations, so all
  pairs in even the largest sequence are seen within 63 epochs, while retaining sequence balance;
- retain 300 epochs, batch 16, 160 steps/epoch and 48,000 optimizer steps;
- make differentiable box IoU, centroid error, log-area error and worst perimeter error the
  primary geometry loss; retain low-weight cross-modal edge structure, smoothness, fold and
  boundary regularizers;
- apply the original v1 screen and exhaustive thresholds unchanged. The 8-pair-per-sequence
  train/validation screen must pass before the 199,798-pair exhaustive train/validation test runs.

The design is informed by RIFT's radiation-insensitive structural matching, NeMAR's
geometry-preservation principle, XoFTR's visible/TIR-specific sub-pixel matching, RoMa's robust
coarse-to-fine dense correspondence, C2RF's cross-modal commonality mining, and the existing
SuperFusion matcher. Those papers motivate components; they do not constitute evidence that v2
passes Anti-UAV300. Because v1 aggregate validation performance was already observed before v2
was designed, a v2 pass qualifies this engineering gate but is not a pristine independent
confirmation of the research claim. External paired replication remains required for that claim.

Execution completed on 2026-09-16 without changing the frozen protocol. The train-only fit used
all 300 epochs and 48,000 optimizer steps; checkpoint `13dd5c47...1135f` records the frozen v1
initialization hash, cache-manifest hash, source revision and the absence of validation/test
access. The immutable sequence-balanced screen (`9e7727a1...15f4c`) failed: train joint pass was
29.30% and held-out validation joint pass was 22.57%, against 95%. Validation median edge-NCC
gain was 0.593 and 99.81% of frames improved in edge-NCC, but median box IoU was 0.386, median
absolute area-ratio change was 1.357, and only 63.25% passed the inverse-residual threshold. The
screen therefore remained HOLD under its pre-registered computation. Per the frozen decision
rule, the exhaustive 199,798-pair audit was not run and paired generator training remains HOLD.
This result must not be replaced by post-hoc threshold or loss changes; any new intervention
requires a separately frozen protocol.

#### Historical Anti-UAV300 protocol v3 (superseded for qualification)

The v2 screen and loss evaluated different geometric objects. SuperFusion's
visible-to-infrared field assigns each infrared output coordinate a visible input coordinate.
Training scored the native infrared-to-visible coordinate map, but the screen instead tried to
recover infrared coordinates by fixed-point inversion of the perimeter of an axis-aligned visible
box. Nonlinear transformation, axis-aligned enclosure and inversion do not commute. This
mismatch does not by itself identify the cause of low accuracy or make the immutable v2 result pass.

The content-addressed root-cause audit (`9a24eb17...9a565`) reran the exact 8-pair-per-sequence
screen through both paths. The native target-to-source support check reaches 94.30% train and
92.16% validation joint pass, while the legacy inverse-box path reproduces 29.30% and 22.57%.
Mean full-field fold fractions are 0.83% and 0.94%; a synthetic 2.1x affine scale also proves
that the legacy fixed-point solver can fail on a valid noncontractive map. Worst-case overlays are
saved beside the audit. Native and inverse-enclosure scores are different diagnostics, not
interchangeable accuracy estimates. Improved inversion barely changes real-data box agreement.
Neither result establishes physical registration accuracy; v2 remains HOLD.

Protocol v3 is a distinct pre-registered intervention:

- initialize from immutable v2 checkpoint `13dd5c47...1135f` and reuse its verified train-only
  cache without opening validation or test during fitting;
- train SuperFusion's two native backward maps, visible-to-infrared and infrared-to-visible,
  rather than numerically inverting either map;
- retain 300 epochs and 16 rotating pairs per training sequence per epoch; batch 8 gives 320
  steps/epoch and 96,000 optimizer steps;
- average the two geometry-first losses and add explicit off-grid inverse-consistency loss by
  composing the reciprocal dense maps;
- retain the original IoU, centroid, area, coverage and edge thresholds in both directions, and
  additionally require 95% of frames to have mean valid-pixel cycle residual at most 0.01, at
  least 99% positive Jacobian pixels, and at least 90% valid cycle coverage;
- run the immutable 8-pair-per-sequence train/validation screen first. Only a pass runs every
  usable train/validation pair; official test remains unopened and report-only.

This direction convention follows the backward sampler defined by Spatial Transformer Networks
and the upstream SuperFusion implementation. SuperFusion explicitly estimates bidirectional
deformation fields; ICON and GradICON motivate reciprocal-map composition as an inverse-
consistency regularizer. References: [STN](https://proceedings.neurips.cc/paper/5854-spatial-transformer-networks.pdf),
[SuperFusion](https://doi.org/10.1109/JAS.2022.106082),
[ICON](https://openaccess.thecvf.com/content/ICCV2021/html/Greer_ICON_Learning_Regular_Maps_Through_Inverse_Consistency_ICCV_2021_paper.html),
and [GradICON](https://openaccess.thecvf.com/content/CVPR2023/html/Tian_GradICON_Approximate_Diffeomorphisms_via_Gradient_Inverse_Consistency_CVPR_2023_paper.html).
Because prior validation aggregates informed v3, even a v3 pass is an engineering qualification,
not pristine independent confirmation; an external paired replication is still required.

**Correction:** v3 also uses inconsistent endpoint/pixel-centre geometry for images versus
points and cycles. Its historical gate is not a current authorization. The separately versioned
[v4 protocol](registration_v4.md) fixes this mismatch without rewriting completed artifacts or
the running v3 process. It uses stronger target-local engineering checks and keeps independent
correspondence evidence and generator eligibility on HOLD. New thresholds are prospective
engineering choices, not tuned passes or evidence of scientific accuracy.

## Label transfer audit (F4)

Labels are carried across the generation step. If generation displaces or deforms object
boundaries, label noise enters silently and is attributed to the data rather than the pipeline.
Before any training run, `aero_ir.data.labels.audit` reports, per generated image:

- IoU between the source box and a box re-estimated on the generated image
- centroid displacement in pixels and as a fraction of box diagonal
- change in target pixel area
- fraction of boxes whose target-background contrast drops below the detectability floor

Images failing the configured thresholds are excluded and counted. The exclusion rate is
reported alongside every result.

## Metrics

**Primary.** mAP@0.5, mAP@0.5:0.95, mAR@0.5, mAR@0.5:0.95, and per-class AP / AR, all from one
pinned COCO evaluator. Direct comparisons are made only within the same split, category mapping
and evaluator version.

**Effect size.** `dAP = AP(real + gen) - AP(real)` at matched budget. Seeds are paired across
arms. Three seeds provide screening estimates only. A pilot estimates the variance and powers
the confirmatory contrasts; percentile bootstrap intervals are used only with at least 10
matched seeds. This, not raw AP, is the quantity every hypothesis is about.

**Stratified (F5).** AP decomposed by target pixel area (COCO small / medium / large plus a
tiny bucket below 16 px^2), by target-background contrast quartile, by thermal polarity
(hot-on-cold vs cold-on-hot), and by background clutter level.

**Fidelity.** FID, LPIPS, SSIM, SDQM and CCDM alongside the full RFS vector, so H1 and H2 are
tested on the same runs rather than in separate studies.

**Deployment.** Latency and mAP after ONNX export and INT8 quantisation, on a fixed device
profile.

## Reproducibility contract

Before a result is reportable, each run must write a content-addressed manifest containing the
config hash, git SHA and dirty state, dataset checksums, seed, resolved environment, hardware,
replay command and complete metric set. `make verify RUN=<manifest.json>` verifies manifest,
config and artifact integrity. `make replay RUN=<manifest.json>` explicitly re-executes the
command from a trusted local manifest into an isolated temporary output root and compares
recorded metrics within the preregistered tolerance. The FLIR runner implements this lifecycle
and its evaluator persists predictions plus mAP@0.5:0.95, mAP@0.5, mAR@0.5:0.95, mAR@0.5 and
per-class AP/AR. GPU training still has to populate and replay the contract before a result is
reportable. Both 128/64-image one-epoch FLIR YOLOX smokes, including the batch-8 x 8-step
accumulation check, are separately content-addressed engineering evidence and their AP must not
enter an E1 table. The bounded full-data timing gate passed with batch 32 x accumulation 2: 160
microbatches, 80 optimiser steps, six multiscale sizes, finite losses, 19.21 measured images/s
including cold-size startup, and 9,827 MiB peak allocation. The clean-snapshot 300-epoch real-only
baseline then completed in 5.987 hours with 11,104 MiB peak allocation and final
mAP@0.5:0.95/mAP@0.5 of 0.3513/0.5757. Its manifest, predictions, checkpoints and complete COCO
metrics verify against commit `fc51f66...775220`. YOLOX's optional L1 regression term was zero
during mosaic training by design and nonzero from displayed epoch 285, when the final
no-augmentation phase began.

The 2026-09-07 full replay completed but failed the frozen 0.002 tolerance: its
mAP@0.5:0.95 was 0.35744 versus 0.35131 originally, and its mAP@0.5 was 0.58947 versus 0.57572.
The input, sampler, optimiser-step and multiscale schedules match, but losses differ from the
first logged interval. Upstream YOLOX 0.3.0 initializes each augmentation worker from
`uuid.uuid4()`, so the declared run seed did not control mosaic, mixup and flip randomness. The
v1 result and failed replay remain immutable evidence; the tolerance must not be relaxed after
observing the difference. The adapter now uses deterministic per-worker seeds, disables cuDNN
benchmarking, enables deterministic Torch algorithms, and freezes cuBLAS/Python hash settings.
Twin short GPU runs must match before preparing a clean v2 baseline and replay.
Historical read-only verification resolves changed tracked inputs from the manifest's recorded
clean Git commit and reports them explicitly; replay execution is refused unless the checkout
itself matches every bound source file.

Separately, the FLIR official map has now been frozen as 3,749 `video_test` pairs, but it cannot
supply generator training data: shared
track/category evidence covers only one of eight sequences and has normalised centre-residual
p95 0.0638 against the frozen 0.02 limit. The three-arm screen therefore remains on hold until a
training-authorised, sequence-disjoint paired source passes registration and is frozen in its own
manifest.
