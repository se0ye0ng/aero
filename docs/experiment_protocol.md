# Controlled experiment protocol

Every factor below is an explicit dimension of the run grid. Nothing is fixed by convention.

## Factors

| Factor | Levels | Rationale |
|---|---|---|
| `train.pretrained` | `false`, `true` (COCO) | **F1.** Random initialisation maximises the penalty for distribution shift. Production training never starts there. This is the single most likely cause of the reported negative result |
| `mixing.budget_mode` | `fixed_total`, `additive` | **F3.** Under `fixed_total`, raising the generated ratio removes real data. Under `additive`, real data is held constant and generated data is added. The two answer different questions and imply opposite conclusions |
| `mixing.gen_ratio` | 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 | resolution around the reported inflection near 0.2 |
| `data.coverage_split` | `same_scene`, `held_out_scenario` | **F2.** `same_scene` reproduces the reported setup, where generation adds no coverage. `held_out_scenario` withholds a scenario slice from the real training set and lets generation cover it — the condition synthetic data actually exists for |
| `curation.policy` | `none`, `random`, `fid_topk`, `rfs_topk`, `marginal_ap` | H4. Compared at an equal generated-sample budget |
| `sensor.model` | `none`, `eo_ir_default`, `matched` | N2. `matched` fits the sensor model to the real set before applying it to generated data |
| `data.domain` | `flir_urban`, `antiuav_small` | generality across regimes. The small, low-contrast regime is where operational IR detection actually lives |
| `detector.arch` | `yolox_s`, `yolox_tiny`, one modern detector | `yolox_*` preserves comparability with the reported result. The third is a generality check only |
| `seed` | 3 seeds | every reported cell is a mean with a spread |

## Experiments

### E1 — reproduce
Fix `pretrained=false`, `coverage_split=same_scene`, `curation=none`, `sensor=none`,
`domain=flir_urban`. Sweep `gen_ratio`. **Success = reproducing the reported negative trend.**
Without this, nothing downstream is interpretable.

### E2 — pretraining ablation (the sign-flip test)
E1 grid with `pretrained` crossed in. Primary readout: does the sign of
`dAP = AP(real + gen) - AP(real)` change with initialisation alone?

### E3 — ratio x budget
Cross `gen_ratio` with `budget_mode`. Separates "generated data is harmful" from
"replacing real data with generated data is harmful". These are not the same claim.

### E4 — curation
Fixed generated-sample budget, varying `curation.policy`. Tests H4 and, by comparing
`fid_topk` against `rfs_topk`, tests H1 and H2 directly.

### E5 — small-target regime with a real coverage gap
`domain=antiuav_small`, `coverage_split=held_out_scenario`. The condition under which
generated data has something to contribute. Highest-value cell in the grid.

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

**Primary.** mAP@0.5, mAP@0.5:0.95, mAR@0.5, mAR@0.5:0.95, and per-class AP / AR — the same
metric set used by the reported result, so numbers are directly comparable.

**Effect size.** `dAP = AP(real + gen) - AP(real)` at matched budget, with a bootstrap CI over
seeds. This, not raw AP, is the quantity every hypothesis is about.

**Stratified (F5).** AP decomposed by target pixel area (COCO small / medium / large plus a
tiny bucket below 16 px^2), by target-background contrast quartile, by thermal polarity
(hot-on-cold vs cold-on-hot), and by background clutter level.

**Fidelity.** FID, LPIPS, SSIM alongside the full RFS vector, so H1 and H2 are tested on the
same runs rather than in separate studies.

**Deployment.** Latency and mAP after ONNX export and INT8 quantisation, on a fixed device
profile.

## Reproducibility contract

Each run writes a manifest: config hash, git SHA, dataset checksums, seed, resolved
environment, hardware, wall-clock, and the complete metric set. `make verify RUN=<id>`
re-executes from the manifest and diffs the metrics. A run that does not reproduce within
tolerance is marked and excluded from reported aggregates.
