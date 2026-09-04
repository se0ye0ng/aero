# References

**This file is load-bearing.** Every statement in this repository about what previous work
found must point at an entry here. A claim without a reference beside it is a claim this
repository does not make.

Nothing in this repository derives from non-public material. If a finding cannot be traced to
a public source, it is not cited, not reproduced, and not used as a baseline.

## Reproduction target for E1

E1 exists to reproduce a **published, citable** negative result before reinterpreting it.
Fill this in before running E1, and mirror it into
`configs/experiment/e1_reproduce.yaml:reproduction_target`.

| Field | Value |
|---|---|
| Citation | TODO |
| Venue / year | TODO |
| Dataset and split | TODO |
| Detector and initialisation | TODO |
| Reported trend | TODO |
| Reported magnitude | TODO |
| Tolerance for "reproduced" | TODO |

If no single published study matches closely enough to reproduce, say so in the paper and
state what E1 reproduces instead — a documented public recipe under stated conditions. An
honest "no exact prior exists, so here is the controlled baseline" is stronger than an
unsourced appeal to what is "widely reported".

## Generation

| Key | Work | Used for |
|---|---|---|
| `diffv2ir` | Visible-to-infrared diffusion with vision-language conditioning (2025) | generator adapter, `configs/generator/diffv2ir.yaml` |
| `pid` | Physics-informed diffusion for infrared image generation, Pattern Recognition (2025) | temperature / emissivity / reflected-radiance conditioning, `configs/generator/pid_tev.yaml`. **Evaluated on SSIM / PSNR / LPIPS / FID only — no downstream task.** This gap is the opening this repository works in |
| `physir-splat` | Physically consistent thermal infrared radiative transfer in 3D Gaussian splatting, CVPR (2026) | N3 background, `src/aero_ir/scene3d/plan.md`. Reconstruction quality, not detection utility |

## Synthetic-data utility and curation

| Key | Work | Used for |
|---|---|---|
| `sdqm` | Synthetic data quality metric for object detection dataset evaluation (2025) | baseline for RFS; domain-agnostic, no sensor physics |
| `marginal-ap-curation` | Online data curation for object detection via marginal contributions to dataset-level AP (2025) | `MarginalAPSelector` baseline |

## Detection and data

| Key | Work | Used for |
|---|---|---|
| `yolox` | YOLOX (public release) | detector; upstream defaults are the training recipe |
| `flir-adas-v2` | Teledyne FLIR ADAS Thermal Dataset v2 | reproduction anchor |
| `llvip` | LLVIP aligned visible-infrared pairs | optional domain |
| `dronevehicle` | DroneVehicle aerial RGB-IR detection | optional domain |
| `antiuav410` | Anti-UAV410 thermal UAV tracking benchmark | small-target regime |
| `cst-antiuav` | CST Anti-UAV, ICCV workshop (2025) | tiny-target regime |

## Format

Add a BibTeX file (`docs/references.bib`) as entries are confirmed, and cite by the keys above
throughout `docs/` and in code comments. Keys, not prose descriptions, so a reader can check.
