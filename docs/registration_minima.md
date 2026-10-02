# MINIMA-XoFTR: fixed-protocol checkpoint comparison

## Motivation and scope

The earlier control-selection and RGB-resolution changes did not improve
Anti-UAV joint box-proxy passes. This test instead changes pretrained cross-modal
features while preserving the existing network, input preparation and geometric
postprocessors. It is not another 300-epoch training run.

[MINIMA](https://github.com/LSXI7/MINIMA) trains matching models using synthetic
cross-modal pairs derived from RGB matching data; see the
[CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html).
That motivation makes its released XoFTR checkpoint a relevant untested candidate,
not evidence that it works on our Anti-UAV sequences. MatchAnything was also found
in the literature search but is not implemented or evaluated in this experiment.

## Acquisition and compatibility

- MINIMA repository pin: `796e7721174f9f829b79b3702bf8c2ae9a3d447a`.
- [Official weight](https://github.com/LSXI7/storage/releases/download/MINIMA/minima_xoftr.ckpt):
  release asset230220857, 44,484,144 bytes; downloaded without overwriting any model.
- Observed SHA256: `551eaa814713b3c60ca380eda022be9b7609e51737302e12370f8943a1e864d7`.
  The release API supplied no digest. Initial acquisition trusts official HTTPS;
  this is our recorded digest, not an independently published author checksum.
- The checkpoint contains a `state_dict` with247 entries. Loading uses
  `weights_only=True`, CPU placement and strict parameter matching through the
  existing adapter. The upstream model's prefix-removal logic handles `matcher.`.
- MINIMA's XoFTR configuration is byte-identical to the pinned configuration already
  used here (SHA256 `4360ea2108cd92a3eb4bb7afab608cd777370d97146b46d795ca8452886bb1eb`).
- MINIMA's XoFTR submodule points to `e9635d8baf95b5731bb1a142eff9a479a99e1e3b`.
  The [upstream comparison](https://github.com/OnderT/XoFTR/compare/e9635d8baf95b5731bb1a142eff9a479a99e1e3b...e0fbea431b30be9742effbf5577c90aa8eb938f9)
  shows one subsequent change: `np.float` to `np.float32` in `src/utils/data_io.py`.
  Our adapter bypasses that wrapper; matching-network code is unchanged.

The official loader/config/license and revision-comparison response are retained
under ignored `experiments/external/minima_796e772/`. They are not vendored into
the project or redistributed with model weights. The downloaded official loader
is inspected, not executed. No model dependencies or installed torch version change.
Bytecode-cache isolation follows the [resolution diagnostic](registration_rgb_resolution.md).

## Frozen experiment

Configuration: `configs/experiment/registration_minima_cpu.yaml`.

1. Verify the previous local-warp and global-geometry reports, dataset files,
   matching code, model configuration and candidate checkpoint identities.
2. Evaluate the same four UAV-TIRVis pairs at the original640-long-side inputs,
   thermal-first inference order and pixel-center coordinate restoration.
   Apply both prior TPS control policies without tuning. Reference coordinates
   enter only scoring after image-only fitting; unsupported points remain failures.
3. Evaluate the same16 Anti-UAV train midpoints, RGB640×360 and IR640×512,
   with the same72/104-row header crops and restored output offsets. Native and
   resized frame hashes must match the original panel. No GT-guided crop or match
   selection, no validation/test access, no new confidence threshold search.
4. Score both TPS policies and the already-defined similarity/affine/homography
   fits from unique reciprocal matches. The latter retain the original split-fit/
   check protocol, analytic inverse and rejection conditions. Report every family,
   including failures, rather than choosing the most favorable model afterward.
5. Run a same-modality known(+8,-8) shift control and an unrelated first-RGB/
   last-IR control. Neither supplies true cross-modal physical correspondences.

This is a fixed-protocol checkpoint swap, not an exact reproduction of the
authors' benchmark. Our preprocessing and empty-fine-match placeholder guard
are held identical to the prior baseline, not changed to chase either model's
best published settings. Full-image background references and target-box proxies
answer different questions and cannot authorize each other. Both panels have
already been examined, so this is exploratory, not unseen-data confirmation.

## Reproduction

With the pinned official asset present and the original local baseline artifacts:

```bash
env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/aero-matplotlib \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -u -m scripts.probe_registration_minima \
  --out-dir experiments/registration_minima_cpu_repeat01
```

Use a fresh output directory. The probe downloads nothing and refuses identity
mismatches. No GPU launch is needed for this panel. Qualification and generator
eligibility remain unchanged unless subsequently established by appropriate
evidence; a completed script or more raw matches is not an approval.

## Completed results — 2026-09-24

All four external pairs and16 Anti-UAV pairs completed on CPU in199.11 seconds
after model loading. The preceding protocol was recorded before aggregate-score
inspection. The candidate loaded strictly without parameter-shape/key failures.

### External reference points

Equal-pair mean PCK uses the original thermal-file pixels. Unsupported landmarks
remain failures, and the same four previously inspected pairs are used throughout.

| TPS selection | Original XoFTR PCK3 | MINIMA-XoFTR PCK3 | Original PCK5 | MINIMA PCK5 |
|---|---:|---:|---:|---:|
| Grid | 84.07% | 80.97% | 87.00% | 87.06% |
| Farthest-point | 80.17% | 86.41% | 89.50% | 92.03% |

The checkpoint effect depends on the fixed control policy; it is not uniformly
positive. The higher FPS PCK3 does not establish fine physical accuracy on Anti-UAV.

### Anti-UAV box-corner proxies

All values below are both-direction IoU>=0.6 counts over the same16 input-header-
cropped pairs. Global models use the original half-fit/half-check reciprocal-match
protocol; TPS uses the original local-consensus protocol. These are separate
postprocessors, not a single combined qualified registration method.

| Postprocessor | Original XoFTR | MINIMA-XoFTR |
|---|---:|---:|
| TPS, grid controls | 1/16 | 1/16 |
| TPS, farthest-point controls | 1/16 | 1/16 |
| Global similarity | 3/16 | 3/16 |
| Global affine | 3/16 | 3/16 |
| Global homography | 2/16 | 2/16 |

The passing pairs are unchanged for TPS, similarity and affine. Homography gains
one pair and loses another, leaving the aggregate unchanged. No failed pair is
dropped. A fitted matrix's analytic inverse or a successful box proxy is not a
measurement of same-surface pixel correctness or a full v4 engineering pass.

The known-shift control yields2,317 matches, all within3 input pixels of the
synthetic displacement. The unrelated-pair control yields one match (versus seven
in the original input-header-cropped XoFTR screen). That single negative control
is not a calibrated precision estimate or evidence that the real pairs qualify.

**Decision:** do not substitute this checkpoint for the current registration
model or launch long training from this result. It improves one external-data
configuration but does not increase Anti-UAV joint box-proxy coverage. Repeating
the same architecture with these weights is not a demonstrated repair. No
registration/generator approval follows, and the full project goal remains open.

Report: `experiments/registration_minima_cpu_01/report.json`.
SHA256: `77babd8f1cef920ed312595d9085bc67bbd4d8ec3030f2565cbf2009af41affa`.
All155 recorded input/source hashes and all saved correspondence-array hashes were
rechecked. The saved global-baseline source hash still matches its current source.
The complete710-test CPU suite, repository Ruff and `git diff --check` passed.
No GPU training, dataset mutation, threshold change or Git push occurred.
