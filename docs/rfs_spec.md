# Radiometric Fidelity Score (RFS)

RFS is a **diagnostic vector**, not a single number. A scalar would hide exactly the
information the study needs: *which* physical property the generator failed to preserve.

Each component is computed on a set of images with labels, producing a distribution.
Generated and real sets are compared by a distributional distance
(1-D Wasserstein by default; MMD and two-sample KS available).

## Components

| # | Component | Definition | Why a detector cares |
|---|---|---|---|
| R1 | Target-background differential `dT` | median intensity inside the box minus median in a dilated annulus around it, per object | the primary detection signal. If the generator compresses `dT`, targets become undetectable regardless of how realistic the image looks |
| R2 | Thermal polarity | sign of R1, aggregated per class | polarity inversion (cold target on warm ground vs the reverse) is a systematic labelling-consistency failure that AP alone cannot expose |
| R3 | Target SNR | R1 divided by the local background standard deviation | separates "low contrast" from "high noise" — two failure modes with different remedies |
| R4 | Radial power spectrum | azimuthally averaged log power spectral density of the image | captures optical blur and over-sharpening. Generators routinely produce edges no IR optic can form |
| R5 | Noise PSD | PSD of the high-pass residual after a matched low-pass | generated images are typically too clean; a detector trained on them fails on real sensor noise |
| R6 | Target pixel-area distribution | box area in pixels, per class | the small-target regime is where operational detection lives and where generators are least faithful |
| R7 | Dynamic range and histogram shape | percentile spread, entropy, clipped-pixel fraction | AGC and 8-bit quantisation reshape the intensity distribution the detector sees |
| R8 | Fixed-pattern / column-structure energy | ratio of column-wise to row-wise variance in the high-pass residual | real focal-plane arrays leave residual non-uniformity; generated images do not |

## Scalar aggregate

For ranking and for curation, components are combined as a weighted sum of normalised
distances:

```
RFS = sum_i w_i * d_i(P_gen, P_real) / d_i_ref
```

where `d_i_ref` is the distance between two disjoint halves of the *real* set — the
irreducible sampling floor. `RFS = 1` therefore means "as far from real as real is from
itself". Weights default to uniform; `configs/rfs/` allows re-weighting, and the weight
sensitivity of every conclusion is reported.

## What RFS is tested against

RFS is only interesting if it predicts something. The evaluation is: across all runs in
E3 and E4, regress `dAP` on (a) FID, (b) LPIPS, (c) the RFS scalar, and (d) the RFS vector.
Report Spearman correlation and cross-validated predictive R^2 for each. **H2 requires that
(c) and (d) beat (a) and (b).** If they do not, that is the paper's result.
