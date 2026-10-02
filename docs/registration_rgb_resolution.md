# Fixed-panel RGB-resolution diagnostic

This follows the negative [control-selection test](registration_local_warp_diagnostic.md).
More control points produced more fitted warps but did not improve Anti-UAV joint
box-corner proxy passes. This test changes matching input rather than training
longer, weakening the qualification gate, or fitting a warp to target annotations.

## Protocol

`configs/experiment/registration_rgb_resolution_cpu.yaml` freezes the preceding
control-selection report by SHA256 and inherits its hashed matching panel and
dependencies. All16 previously examined train pairs are retained. No validation
or test frames are accessed, and no new human labels are requested.

- Baseline: RGB640×360 and native IR640×512, with72 and104 top rows removed.
- Candidate: RGB1280×720 directly resized from the decoded1920×1080 native frame;
  IR remains640×512. Remove144 RGB rows and104 IR rows, preserving header extent.
- Use the same pinned XoFTR640 checkpoint and unchanged confidence thresholds.
  No target-box crop, target-box matching restriction or geometric refit to GT.
- Decode the exact frozen frame IDs and require matching native-image byte hashes.
- Restore predictions to the same baseline working grids using pixel-center
  resize coordinates: `(p + crop_offset + 0.5) / scale - 0.5`. Annotation box edges
  retain the original edge-scaling convention. These are different operations.
- Apply both previously declared local-consensus TPS control policies, grid and
  farthest-point, without changing filtering, support rejection or thresholds.
  Report both, rather than selecting whichever produces a favorable result.
- Score both directional box-corner IoUs and joint passes on all16 pairs. These
  remain annotation proxies, not physical pixel accuracy or the full v4 gate.

The first pair also reruns both original640-input matching directions and compares
point coordinates/confidences to the saved baseline (absolute tolerance1e-4,
identical array shapes). A same-modality known(+8,-8) high-resolution shift and
an unrelated first-RGB/last-IR pair are implementation/control checks. A match on
an unrelated pair is not necessarily numerical malfunction; it cannot be treated
as evidence of true same-frame physical correspondence.

The candidate changes feature scale **as well as** retained source detail, so it
does not isolate information loss from scale sensitivity. A negative result would
not prove that higher native detail can never help another model. Central HUD
remains unmasked. No result here alone authorizes a generator.

## Source/cache integrity

The vendor commit must match the original pin. Tracked modifications and untracked
non-bytecode files are refused. Existing untracked `__pycache__/*.pyc` files are
preserved but never used: before loading the matcher, Python bytecode lookups are
redirected to a fresh empty temporary directory and bytecode writes are disabled.
The ignored bytecode paths are recorded separately; they are not described as a
fully clean checkout. Weight and input/source hashes are checked before and after
inference. No external code or weights are downloaded during the run.

## Reproduction

Use a fresh output directory; the original experiments are never overwritten:

```bash
env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/aero-matplotlib \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -u -m scripts.probe_registration_rgb_resolution \
  --out-dir experiments/registration_rgb_resolution_cpu_repeat01
```

## Completed result — 2026-09-24

Actual report: `experiments/registration_rgb_resolution_cpu_01/report.json`.
SHA256: `1c11080c556088bd161f28bf08db8f86032ad9e7a17b0d412071644d1d243987`.
All 16 pairs completed in 353.99 seconds after model loading. The protocol above
was written before inspecting the candidate's aggregate scores. The first-pair
bidirectional baseline replay passed.

| Selection policy | RGB width | Forward fits | Reverse fits | Joint box-corner passes |
|---|---:|---:|---:|---:|
| Grid | 640 | 6/16 | 5/16 | 1/16 |
| Grid | 1280 | 4/16 | 3/16 | 1/16 |
| Farthest-point | 640 | 8/16 | 9/16 | 1/16 |
| Farthest-point | 1280 | 6/16 | 5/16 | 1/16 |

The same pair passes jointly in all four conditions; there are no new passing
pairs or lost passing pairs. Higher-resolution input has both-box matches in
11/16 forward and 10/16 reverse cases, unchanged from the lower-resolution panel.
Raw match counts and fit counts therefore must not be substituted for successful
target alignment. All missing/unsupported/incorrect cases remain in the denominator.

The known-shift control has 10,337 matches, all within three high-resolution input
pixels of the true synthetic displacement. This verifies a same-modality control,
not cross-modal accuracy. The unrelated first-RGB/last-IR control still returns
31 matches, further ruling out raw count alone as an accuracy certificate.

Decision: **do not adopt this input-resolution change or launch long training
based on it**. The previously measured support and alignment problems persist.
This does not prove that resolution is irrelevant for every matcher or that no
matching solution exists; source detail and feature scale change together here.
Local TPS and these matching proxies still do not qualify physical registration
or authorize paired generator training. Investigation must target a different
mechanism rather than repeat the same enlargement or sampling-only experiment.

Verification: 708 complete CPU tests, repository Ruff and `git diff --check`
passed. All 111 report input/source hashes and saved match-array hashes were
rechecked. These are implementation and artifact-integrity checks, not scientific
qualification. No GPU training, source dataset modification or Git push occurred.
