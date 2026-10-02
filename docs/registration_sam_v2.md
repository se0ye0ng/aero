# SAM ROI coordinate repair (diagnostic v2)

This is **SAM diagnostic v2**, not the historical registration-v2 trained model.
Human reviewers are not prerequisites. No qualification threshold, existing result,
dataset pairing, model checkpoint, or reviewer response has been overwritten.

## Confirmed implementation problem

The original ROI helper clipped its requested square to image bounds and resized
the remaining rectangle to256x256. This introduced unequal x/y scales and moved
the target away from the crop centre. The silhouette search nevertheless started
from identity with only±8px translation,±10degree rotation and scale0.9–1.1.

For `20190925_222534_1_4`, the old RGB crop was330x203 native pixels. Its box centre
became(127.89,208.21); IR became(128.88,128.88). The initial y difference was79.33
crop pixels. This is a preprocessing/initialization mismatch, not proof of SAM failure.

The original v1 scripts and outputs are retained unchanged for reproducibility.

## Corrected implementation

- `src/aero_ir/registration/roi_geometry.py`: isotropic affine native→crop mapping.
  Both axes use `s=256/max(3*max(box width,box height),48)`. The box centre maps
  exactly to(127.5,127.5), even at camera boundaries. Missing image support is
  represented explicitly; padding and heuristic HUD never count as evidence.
- `src/aero_ir/registration/silhouette_search.py`:625 coarse similarities at64px,
  two retained candidates, then local refinement at128/256/256px. Pixel-centre
  resize conjugation and exact nearest-neighbour mask resizing are used. The same
  search applies to paired and shuffled cases. It is bounded, not a global optimum
  certificate. Angle/translation edge and out-of-initial-scale warnings are recorded.
- The point transform and its analytic inverse are both saved, including composition
  into original RGB/IR camera pixel coordinates. Foreground falling into missing
  support is rejected in both directions. Invertibility alone is not physical accuracy.
- Selection still uses the v1 objective, symmetric Dice loss + Chamfer/crop size.
  **A lower combined objective can worsen Chamfer.** This is explicitly reported,
  not silently treated as two-metric improvement or removed from the panel.

## Completed CPU replay

Authoritative corrected replay:
`experiments/registration_sam_v2_cached_repair_02/report.json`.
The earlier`_01` is retained as an intermediate run before exact-nearest coarse-grid
resampling, deterministic tie-breaking, and expanded diagnostics were finalized.
The`_02` result includes copies/hashes of executed sources and all32 mask artifacts.

This replay reprojects saved SAM masks through the old and new coordinate transforms.
It **does not rerun SAM**, recover lost native image detail, or fix semantic mask
mistakes. Original v1 mask eligibility is preserved, including all abstentions.

| Group | Total | Foreground-safe fits | Valid identity baseline and fit | Median Dice loss before→after, same cases | Median Chamfer before→after, same cases |
|---|---:|---:|---:|---:|---:|
| Actual pairs | 16 | 10 | 9 | 0.247674→0.160190 | 2.979718→1.395751 |
| Shuffled sequences | 16 | 9 | 9 | 0.436291→0.331692 | 5.791768→5.577252 |

Lower is better; distances are256px corrected-crop units, not native physical
correspondence error. Nine of nine comparable actual pairs improve Dice, but so
do all nine shuffled pairs. For the nine common RGB sources, actual-pair final
Dice loss is lower in seven cases, not all nine. This does not establish a qualified
pixel correspondence signal. One actual pair's Chamfer worsens despite lower Dice.

There are no selected transforms at the hard parameter limits; however one actual
pair is near the explored translation edge, one near the angle edge, and two use
scales beyond the initial coarse scale grid. Do not describe this as elimination
of all search-limit effects. One actual case lacks a foreground-safe identity
baseline, so it is not included in the nine-case before/after medians.

The problem boundary case was also checked with **actual SAM CPU inference on
one newly prepared RGB crop**, using the pinned pretrained weights. The crop centre
was correct, valid support was61.33%, and all three prompt masks were produced;
prompt-jitter IoUs were0.9258/0.9239. This is one-image integration smoke evidence,
not a completed fresh sixteen-pair experiment.

## Completed fresh GPU extraction

Report: `experiments/registration_sam_train16_v2/report.json` (SHA256
`99c5011cf00a5efe3d54bc0ca04baac3b03cff1d4022b0d90c90e6acb5ccb205`).
The run used CUDA and took73.47 seconds of reported extraction/search time.
All38 artifact hashes, six source hashes, and the parent report hash were checked.
All32 crops have isotropic scaling and map box centres to(127.5,127.5).
All58 available baseline/selected score components recomputed exactly.

| Group | Panel | Safe fits | Comparable before/after | Median Dice loss before→after | Median Chamfer before→after |
|---|---:|---:|---:|---:|---:|
| Actual pairs | 16 | 8 | 7 | 0.244876→0.161115 | 2.949863→1.336335 |
| Shuffled sequences | 16 | 7 | 7 | 0.566447→0.264449 | 5.622722→3.572008 |

Only10/16 RGB and12/16 IR mask sets pass the unchanged stability/support heuristics.
The two previously fit pairs lost eligibility because RGB prompt-jitter IoU was0.7122
for`20190925_131530_1_1`, and IR prompt-jitter IoU was0.7854
for`20190925_141417_1_2`, both below the fixed0.8 threshold. Do not compare medians
across changing eligible subsets as evidence of overall improvement.

Actual-pair Dice is better than shuffled-pair Dice on5/6 common RGB sources.
All seven comparable actual pairs and all seven comparable shuffled pairs improve
Dice. Actual pair`20190925_210802_1_7` worsens Chamfer from7.3609 to17.7582.
Perturbed prompts preserve Dice improvement on14/14 comparable actual observations,
but two worsen Chamfer; these are correlated pseudo-masks, not independent accuracy.
This completes the coordinate correction experiment, not registration qualification.

The subsequent [matched-candidate non-deterioration ablation](registration_sam_pareto.md)
prevents the observed Chamfer regressions but does not establish correct correspondences.

### Reproduction command

```bash
cd /lustre/winston1214/project/aero
AERO_ANTIUAV300_ROOT=/lustre/winston1214/dataset/Anti-UAV300 \
  bash scripts/run_registration_sam_v2.sh
```

Output: `experiments/registration_sam_train16_v2/report.json`.
The wrapper reuses the already installed SAM1.0 and pinned ViT-B weights
(`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`).
No downloads, installation, hard-coded GPU index or overwrite. It preserves the
scheduler's GPU visibility, runs CPU regression tests, checks CUDA, extracts32
new mask sets with original and perturbed box prompts, then searches both real
and shuffled pairs. RGB remains grayscale-repeated exactly as in v1; changing
color treatment is not mixed into this coordinate repair.

Use a fresh `AERO_SAM_V2_OUTPUT` for a repeat. The v1 report and its32 artifacts
must remain available for provenance. Fresh eligibility may differ from the replay;
report complete-panel coverage and common eligible cases, not only favorable cases.

The fresh full-panel run is complete. No300-epoch training or generator job is
authorized by finishing this diagnostic. Nighttime light-only RGB masks versus
thermal body masks remain a known potential semantic mismatch; stability or a
better optimized silhouette score cannot by itself resolve it.
