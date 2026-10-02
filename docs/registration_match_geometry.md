# Geometry fitted to observed detector-free matches

This CPU diagnostic tests whether low-degree-of-freedom transforms explain saved
XoFTR correspondences. It does not project the previous v6 flow into an affine map:
it fits actual image-derived correspondences, so a failed old flow approximation
does not settle this alternative.

```bash
.venv/bin/python -B -m scripts.probe_match_geometry \
  --report experiments/xoftr_overlay_train16_01/report.json \
  --output-dir experiments/registration_match_geometry_01
```

The source report must be train-only. The script verifies saved NPZ hashes and uses
the same frames for `raw_postfilter` and `input_header_crop`; it does not read videos,
train weights, access validation/test or launch GPU inference. Boxes are evaluation
proxies only and never select fitting matches.

- Keep matches with reverse consistency within 2 pixels at **both** endpoints.
- Deduplicate joint endpoint pairs rounded to 1 pixel, retaining subpixel coordinates
  for fitting. Fixed RNG0 splits each deduplicated set into fitting/check halves.
- Similarity, affine and homography require at least 6, 8 and 12 total unique pairs.
- RANSAC uses a 3-pixel forward residual, 5,000 iterations maximum and confidence0.999.
  Only the fitting half estimates parameters; the check half is never refitted.
- Reject insufficient/collinear/nonfinite support, failed estimates, singular or
  ill-conditioned transforms, and projective poles crossing either retained domain
  or evaluation box. Failed fits stay failed: there is no identity fallback.
- Report forward and analytic-inverse box enclosure IoU, held-out residuals in each
  direction, and retained-domain numerical support on a 32x32 interior lattice.

The analytic inverse trivially addresses algebraic inverse consistency; it is not an
independent correctness measurement. Held-out match residuals test agreement with
the detector-free model's other predictions, **not physical GT**. Numerical image
coverage does not establish observed feature coverage or full-image alignment.
Summary means use valid fits and report eligibility denominators; the number with
both box IoUs>=0.6 uses all frames as the denominator. Missing fits must not disappear
from model comparisons. All qualification and generator claims remain HOLD.

These inputs exclude the header but still contain any central reticle. This diagnostic
cannot independently distinguish a true shared scene structure from matching HUD.

## Executed train16 result

Saved report: `experiments/registration_match_geometry_01/report.json`.

| Input condition | Transform | Eligible fits /16 | Both box IoUs >=0.6 /16 | Mean forward / inverse IoU on eligible fits |
|---|---|---:|---:|---:|
| Raw inference, header matches removed | Similarity | 12 | 3 | 0.417 / 0.389 |
| Header removed before inference | Similarity | 9 | 3 | 0.487 / 0.465 |
| Raw inference, header matches removed | Affine | 12 | 3 | 0.434 / 0.419 |
| Header removed before inference | Affine | 10 | 3 | 0.452 / 0.428 |
| Raw inference, header matches removed | Homography | 6 | 2 | 0.361 / 0.388 |
| Header removed before inference | Homography | 5 | 2 | 0.472 / 0.451 |

Eligible subsets differ, so larger conditional mean IoU does not demonstrate overall
improvement. Cropping does not increase the all-frame count meeting both box-IoU
checks. Homography additionally has poles crossing a retained domain or box on
5 raw-postfiltered and 4 input-cropped pairs. Under input cropping, mean inverse
retained-domain support is 0.772 (similarity), 0.728 (affine), 0.794 (homography).
No tested global transform solves the registration bottleneck on this bounded panel.
This result does not exclude a separately validated local/piecewise transform.
