# Problem statement

## Setting

Infrared target detection is trained on data that cannot be freely acquired. Range, aspect
angle, atmospheric condition and time of day are all bounded by physics and by schedule.
Programmes therefore substitute simulated imagery, and increasingly generative imagery, for
the scenarios they cannot record.

## The observed research landscape

Published downstream effects vary with the dataset, task and mixing protocol. Recent work also
introduces task-aware synthetic-data metrics and task-oriented IR generation. The specific
results and direct competitors are recorded in [`references.md`](references.md). E1 transfers
one named fixed-total mixing protocol to IR; it is not an exact reproduction.

## The claim of this work

The testable claim is narrower: sensor-aware radiometric features may explain additional
variation in detector utility after controlling two properties of the experiment:

1. **A fidelity criterion that may omit sensor-specific variables.** Perceptual and recent
   detection-specific metrics are strong baselines, but they do not explicitly model thermal
   polarity, sensor noise, MTF or fixed-pattern structure.

2. **A varying training protocol.** Initialisation, scenario overlap and whether generated data
   replaces or adds to real data can change the question being answered. They are controlled or
   crossed here rather than assumed to have a particular direction of bias.

## Hypotheses

- **H1** Perceptual fidelity metrics (FID, LPIPS, SSIM) have lower held-out predictive power
  than task-aware baselines for the downstream AP change caused by adding generated data.
- **H2** A radiometric consistency vector (see `rfs_spec.md`) adds held-out predictive power
  over perceptual metrics and recent detection-specific metrics (SDQM and CCDM).
- **H3** The effect of adding generated data interacts with initialisation and budget mode.
  Its direction and magnitude are empirical outcomes, not assumptions.
- **H4** Selecting generated samples by radiometric consistency outperforms random mixing and
  outperforms selection by perceptual fidelity, at an equal generated-sample budget.

Each hypothesis is falsifiable. A null result for H2 or H3 remains reportable within the tested
detectors, generators and domains; it does not establish a field-wide conclusion.

## Non-goals

- Beating the state of the art on any IR detection benchmark. The detector is an instrument,
  not the contribution.
- Proposing a new generative architecture. The contribution is the criterion and the protocol.
- Any use of proprietary imagery, labels, sensor specifications or requirement documents.
  Public data only.

## Related work and the gap

| Work | Contribution | What it does not do |
|---|---|---|
| Visible-to-IR diffusion with vision-language conditioning | high-quality IR translation | no downstream detection evaluation |
| Physics-informed IR diffusion (temperature / emissivity / reflected-radiance decomposition) | physical priors inside the generator | evaluated only on SSIM / PSNR / LPIPS / FID — **no downstream task** |
| Physically consistent thermal 3D Gaussian splatting | radiometric novel-view synthesis | reconstruction quality, not detection utility |
| SDQM and CCDM synthetic-data quality metrics; online curation by marginal AP | utility-aware detection-data evaluation and selection | no explicit EO/IR sensor or radiometric model |
| Task-oriented infrared diffusion | spectral physical consistency plus downstream detection/segmentation | does not establish a generator/domain-held-out radiometric utility predictor |

The proposed position is therefore the **held-out predictive link between sensor/radiometric
statistics and downstream utility**, evaluated against current task-aware baselines. Whether
that position is empirically useful is the central experiment, not a premise.
