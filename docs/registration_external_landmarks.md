# External landmark diagnostic — UAV-TIRVis

This is an additional reference check authorized on2026-09-24, not a dataset
replacement or an Anti-UAV qualification certificate. The original Anti-UAV
objective, thresholds, checkpoints and generator HOLD remain intact.

## Why this reference

The authors release original RGB/thermal images and corresponding landmark text
files. Their manual registration utility exports coordinates on the original
image grids, after scaling display clicks back to full resolution. We use those
previously authored points only for evaluation, not their reference-fitted TPS
images as model input. No new annotation task is assigned to the user.
[Official repository](https://gitlab.upb.ro/etti/dcae-public/arh/research/uav-tirvis)
and [dataset paper](https://doi.org/10.3390/jimaging11120432).

The scene domain is aerial imagery of terrain/buildings, not airborne UAV targets
seen from ground cameras. A positive result cannot establish Anti-UAV tiny-target
accuracy. A single homography may be inadequate for scene parallax. Manual
reference uncertainty is not quantified: the display tool uses at most1000
pixels on the long side and integer coordinate conversion. At4000-pixel RGB
width, one display pixel spans four native pixels; this is quantization scale,
not a complete annotation uncertainty estimate. Do not claim subpixel physical
certification from this reference.

## Frozen screen, before scores

Configuration: `configs/experiment/registration_external_landmarks_cpu.yaml`.

- Official repository commit `b6b914123bfe3074af8683476a1d1929f7a6cdc5`.
- Lowest numeric sample ID in each of Mountain, MountainResort, ResidentialArea
  and Seaside. Exactly one per scene; this is a four-pair pipeline screen, not
  evaluation of the complete published dataset.
- Download only original `V.JPG`, `T.JPG` and their two point files. Verify each
  against its pinned Git blob and retain a SHA256 input manifest.
- Existing frozen XoFTR and LoFTR checkpoints; no fitting or threshold sweep.
- Grayscale inputs, long side at most640, dimensions divisible by8. Restore
  native coordinates with pixel-centre-aware resizing in both modalities.
- Estimate thermal-to-RGB homography from model matches only: OpenCV RANSAC,
  native RGB threshold5px, confidence0.999, at most10000 iterations.
- RANSAC seeds0/1/2 measure estimator variability, **not training-seed spread**.
- Read reference coordinates separately; never use them to fit/select a
  transform, rank matches, choose crops or choose the best seed.
- Report forward RGB and inverse thermal landmark errors. Native thermal
  PCK at1/3/5/10px uses all reference points, including failed projections.
  Report point-pooled and equal-pair macro scores. Finite-only median/p95 are
  explicitly conditional and must not hide failed transforms.

The acquired panel contains241 landmark pairs:44 in Mountain/1,34 in
MountainResort/1,92 in ResidentialArea/1 and71 in Seaside/1. Mountain/1 uses
4032×3024 RGB and1280×1024 thermal files; the others use4000×3000 RGB and640×512
thermal files. Error units refer to these stored image grids, not independently
verified sensor sampling resolution. Pooled pixel thresholds do not define a
common physical angular-error tolerance across scenes. All16 downloaded files
(27,493,086 bytes) matched their pinned upstream Git blobs.

## Commands

From the repository root, acquire the fixed small panel once:

```bash
.venv/bin/python -m scripts.fetch_registration_landmarks
```

Run CPU evaluation into a fresh directory:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m scripts.probe_external_registration_landmarks \
  --device cpu --out-dir experiments/registration_external_landmarks_cpu_01
```

The fetcher does not execute downloaded Python or use unpinned branch inputs.
Existing inputs must match their recorded Git blobs; evaluation refuses an
existing output directory. Interrupted outputs are not completion evidence.

## Release boundary

No explicit dataset redistribution license was identified in the inspected
repository root. Public download availability and the paper's license do not
establish a blanket dataset redistribution grant. Raw images, reference points
and derived visualizations remain in ignored `experiments/`; do not add them to
Git. Any public release needs attribution and a separate data/derivative review.
The project-owned evaluator and aggregate measurements are separate artifacts.

No external result automatically changes any original registration gate. Further
work requires full-panel evaluation, independent-reference quality assessment,
and an explicit Anti-UAV domain-transfer test; it cannot be inferred from this
four-pair screen.

## Executed CPU screen (2026-09-24)

Both frozen models completed all four pairs; all241 reference landmarks were
retained. All estimated transforms were invertible and all reference projections
finite. RANSAC seeds0/1/2 produced identical reported scores. These repetitions
are not independent models and do not satisfy a multi-training-seed efficacy
claim. Results below are a bounded implementation/measurement diagnostic.

| Pair | XoFTR median error, thermal file pixels | LoFTR median error, thermal file pixels |
|---|---:|---:|
| Mountain/1 | 4.2011 | 4.9013 |
| MountainResort/1 | 17.1794 | 3.0720 |
| ResidentialArea/1 | 6.9876 | 6.1415 |
| Seaside/1 | 8.2870 | 7.9866 |

| Measure, mean over the four pairs | XoFTR | LoFTR |
|---|---:|---:|
| Reference points within1 thermal file pixel | 8.7584% | 7.6046% |
| Within3 pixels | 36.1032% | 37.3292% |
| Within5 pixels | 41.2791% | 48.5895% |
| Within10 pixels | 55.1637% | 65.1137% |

The point-pooled3px fractions are37.3444% and35.6846%, respectively, showing why
weighting must be explicit: the ranking differs from equal-pair averaging. These
are landmark PCKs, **not** Anti-UAV frame qualification rates or detection AP.
No successful-method or full-dataset superiority claim is justified.

XoFTR returned1789–3699 matches per pair, yet one transform had17.18px median
reference error. Match quantity alone cannot certify accurate registration.
However, these transform errors conflate correspondence errors, global-homography
model limitations and annotation uncertainty. They do not prove every underlying
match is wrong or establish a unique root cause. Do not promote either matcher
to an automatic GT producer on these results.

Artifacts:

- `experiments/registration_external_landmarks_cpu_01/report.json`, SHA256
  `2d74696136ca4f99efdea4e2f816b3be0742c2e9b9505a56096d8261875a4d6f`.
- Input `manifest.json`, SHA256
  `0b389d397dd5c74174bc6befa2340dfe4747fd44df14b285a3dd281b8579bfb3`.
- Saved per-pair native-coordinate matches, transform matrices, input/source/
  weight hashes and all seed-level measurements permit metric rechecking.
- Recorded evaluation time99.39s excludes interpreter/import startup and data
  acquisition; it is not a throughput benchmark.

Verification:675 CPU tests and repository Ruff checks passed. This includes11
new synthetic tests for coordinate centres, transform direction, invalid-reference
rejection, all-failure denominators, image-match-only estimation and input hashes.
Those synthetic tests check implementation, not real-world qualification.

The next diagnostic should separate geometric model inadequacy from matching
error against these references, before another training launch or a larger
benchmark download. Reference-fitted transforms, if used for that diagnostic,
must be labelled as oracle capacity checks, never image-only method results.

That follow-up has now been executed: [geometry capacity, image-only local TPS,
and actual Anti-UAV transfer](registration_local_warp_diagnostic.md). External
accuracy improves, but the frozen candidate still fails the Anti-UAV transfer
screen; neither result is a generator approval.
