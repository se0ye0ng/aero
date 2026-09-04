# Problem statement

## Setting

Infrared target detection is trained on data that cannot be freely acquired. Range, aspect
angle, atmospheric condition and time of day are all bounded by physics and by schedule.
Programmes therefore substitute simulated imagery, and increasingly generative imagery, for
the scenarios they cannot record.

## The reported result

Studies that mix generated IR into a detector training set report a negative effect: recall
falls monotonically with the generated fraction, precision peaks at a low fraction and then
declines, and the conclusion drawn is that generated data cannot substitute for real data.

## The claim of this work

That conclusion is under-determined by the evidence. It rests on two things that are
properties of the experiment rather than of the data:

1. **A fidelity criterion misaligned with the detector.** Generated IR is judged by perceptual
   similarity — SSIM, PSNR, LPIPS, FID. A detector does not consume perceptual similarity. It
   consumes target-background contrast, thermal polarity, edge energy at the target scale, and
   the noise floor it has to separate signal from. Two images can be perceptually close and
   radiometrically incompatible.

2. **An uncontrolled training protocol.** Random initialisation, generated data drawn from the
   same scenes as the real data, and a mixing ratio confounded with total dataset size each
   independently bias the measured effect toward negative.

## Hypotheses

- **H1** Perceptual fidelity metrics (FID, LPIPS, SSIM) have low predictive power for the
  downstream change in detection AP caused by adding generated data.
- **H2** A radiometric consistency vector (see `rfs_spec.md`) has substantially higher
  predictive power for the same quantity.
- **H3** The sign of the effect of adding generated data flips from negative to positive under
  identified conditions — specifically pretrained initialisation combined with a genuine
  scenario coverage gap.
- **H4** Selecting generated samples by radiometric consistency outperforms random mixing and
  outperforms selection by perceptual fidelity, at an equal generated-sample budget.

Each hypothesis is falsifiable, and a null result for H2 or H3 is still a reportable finding:
it would establish that the negative result is robust to the conditions tested here, which no
published work currently shows.

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
| Synthetic data quality metrics; online data curation by marginal AP contribution | utility-aware data selection | domain-agnostic; no sensor or radiometric physics |

The unoccupied position is the **link between the physics and the downstream utility**.
Physics-informed generation exists. Utility-aware curation exists. Nothing connects them, and
nothing evaluates physically-conditioned IR generation on the task the data is generated for.
