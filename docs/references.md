
# References

**This file is load-bearing.** Every statement in this repository about what previous work
found must point at an entry here. A claim without a reference beside it is a claim this
repository does not make.

Nothing in this repository derives from non-public material. If a finding cannot be traced to
a public source, it is not cited, not reproduced, and not used as a baseline.

> **Verification status.** Entries marked *(verify)* were collected from abstracts and
> summaries and must be checked against the PDF before any number is quoted in a paper or a
> figure caption. Do this once, at the start of Phase 1, and strike the marker.

---

## The state of the question

The premise worth stating clearly: **the published answers disagree, and the disagreement is
unexplained.** That is a stronger motivation than "synthetic data is known to hurt", and it is
what this repository is actually built to resolve.

| Position | Source | What it measured |
|---|---|---|
| Mixing synthetic with real **helps**, with a non-monotonic optimum | `vanherle2022` | fixed-total mixing, industrial objects, Mask R-CNN |
| Synthetic **trails** real in-domain but **beats** it out-of-domain | `mayr2024` | thermal IR segmentation, two datasets |
| Synthetic-trained detectors differ from real-trained ones **layer by layer**, head most | `sundstrom2023` | YOLOv3, CKA analysis |
| Generated IR augmentation **helps** small-target detection | `kim2022`, `hybrid2024` | infrared small target detection |
| Generated image distributions are **less diverse** than real ones | `diversity2023` | Stable Diffusion output statistics |

Note what is missing from every row: **none of them controls initialisation, budget mode and
scenario coverage simultaneously, and none evaluates a physics-conditioned IR generator on a
downstream task.** That is the gap.

---

## Reproduction target for E1

E1 reproduces a **published, citable** controlled-mixing result before reinterpreting it.
Mirror the decision into `configs/experiment/e1_reproduce.yaml:reproduction_target`.

**Recommended anchor: `vanherle2022`.** It is the closest published *controlled* mixing study
— fixed total image count, systematic ratio sweep, transfer learning, AP on real test data —
which is exactly the protocol E1 needs, and it is public and reproducible. E1 transfers that
protocol to infrared.

| Field | Value |
|---|---|
| Citation | `vanherle2022` |
| Venue / year | BMVC 2022 (arXiv:2211.16066) |
| Dataset and split | DIMO (industrial metal objects); E1 transfers the protocol to FLIR ADAS v2 thermal |
| Detector and initialisation | Mask R-CNN + ResNet101, transfer learning; E1 uses YOLOX-s, both initialisations |
| Reported trend | mixing outperforms real-only, with a non-monotonic optimum at a high synthetic-to-real ratio *(verify)* |
| Reported magnitude | best mixed ratio ~80.1 AP vs real-only baseline; fine-tuning ~82.1 AP *(verify)* |
| Tolerance for "reproduced" | the *shape* — a non-monotonic curve with an interior optimum — not the absolute AP, since the domain differs |

**Thermal precedent: `mayr2024`.** Report alongside E1. It is the strongest published evidence
that the sign of the effect depends on the evaluation setting rather than on the data: in-domain
the synthetic-trained model trails the real-trained one, and on an unseen dataset it exceeds it
*(verify)*. This is the published form of the coverage-gap hypothesis that F2 and E5 test.

**If neither reproduces**, say so in the paper and state what E1 reproduces instead — a
documented public recipe under stated conditions. An honest "no exact prior exists, so here is
the controlled baseline" is stronger than an unsourced appeal to what is widely reported.

---

## Synthetic and generated data for detection

| Key | Work | Used for |
|---|---|---|
| `vanherle2022` | Analysis of Training Object Detection Models with Synthetic Data. BMVC 2022. [arXiv:2211.16066](https://arxiv.org/abs/2211.16066) | **E1 reproduction anchor.** Fixed-total mixing protocol |
| `sundstrom2023` | Object Detector Differences when using Synthetic and Real Training Data. SN Computer Science 2023. [arXiv:2312.00694](https://arxiv.org/abs/2312.00694) | layer-wise account of *where* synthetic training diverges; motivates the stratified analysis in Phase 3 |
| `diversity2023` | Diversity and Diffusion: Observations on Synthetic Image Distributions with Stable Diffusion. [arXiv:2311.00056](https://arxiv.org/abs/2311.00056) | diversity collapse as a candidate mechanism; relates to `RFSSelector(mode="coverage")` |
| `sdqm2025` | SDQM: Synthetic Data Quality Metric for Object Detection Dataset Evaluation. [arXiv:2510.06596](https://arxiv.org/abs/2510.06596) | baseline for RFS; domain-agnostic, no sensor physics |
| `marginal-ap2025` | Online Data Curation for Object Detection via Marginal Contributions to Dataset-level AP. [arXiv:2511.14197](https://arxiv.org/abs/2511.14197) | `MarginalAPSelector` baseline |

## Thermal / infrared specific

| Key | Work | Used for |
|---|---|---|
| `mayr2024` | Narrowing the Synthetic-to-Real Gap for Thermal Infrared Semantic Image Segmentation Using Diffusion-based Conditional Image Synthesis. CVPR Workshops (PBVS). [openaccess](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Mayr_Narrowing_the_Synthetic-to-Real_Gap_for_Thermal_Infrared_Semantic_Image_Segmentation_CVPRW_2024_paper.pdf) | **thermal precedent for a domain-dependent sign**; the published form of the F2 / E5 hypothesis |
| `kim2022` | GAN-Based Synthetic Data Augmentation for Infrared Small Target Detection. IEEE TGRS 2022. [IEEE](https://ieeexplore.ieee.org/document/9786867/) | positive result in the small-target regime; E5 counterpoint |
| `hybrid2024` | Infrared Small Target Detection Improvement via Hybrid Data Augmentation Using Diffusion Models and GAN. IEEE 2024 | positive result; generator comparison |
| `irsurvey2024` | A Comprehensive Survey on Synthetic Infrared Image Synthesis. [arXiv:2408.06868](https://arxiv.org/abs/2408.06868) | IR radiometry background; simulation-tool landscape for the `synthetic_baseline` arm |
| `maritime2023` | Thermal-Infrared Remote Target Detection with 3D Game-Based Synthetic Data Augmentation. [arXiv:2310.20412](https://arxiv.org/abs/2310.20412) | prior art for rendered-synthetic IR training data |

## Generation

| Key | Work | Used for |
|---|---|---|
| `diffv2ir2025` | DiffV2IR: visible-to-infrared diffusion with vision-language understanding. [arXiv:2503.19012](https://arxiv.org/abs/2503.19012) | generator adapter, `configs/generator/diffv2ir.yaml` |
| `pid2025` | PID: Physics-Informed Diffusion Model for Infrared Image Generation. Pattern Recognition 2025. [arXiv:2407.09299](https://arxiv.org/abs/2407.09299) · [code](https://github.com/fangyuanmao/PID) | temperature / emissivity / reflected-radiance conditioning, `configs/generator/pid_tev.yaml`. **Evaluated on SSIM / PSNR / LPIPS / FID only — no downstream task.** This gap is the opening this repository works in |
| `physirsplat2026` | PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting. CVPR 2026. [page](https://cvpr.thecvf.com/virtual/2026/poster/37591) | N3 background, `src/aero_ir/scene3d/plan.md`. Reconstruction quality, not detection utility |

## Detection and data

| Key | Work | Used for |
|---|---|---|
| `yolox` | YOLOX (public release) | detector; upstream defaults are the training recipe |
| `flir-adas-v2` | Teledyne FLIR ADAS Thermal Dataset v2 | reproduction anchor |
| `llvip` | LLVIP aligned visible-infrared pairs. [arXiv:2108.10831](https://arxiv.org/abs/2108.10831) | optional domain |
| `dronevehicle` | DroneVehicle aerial RGB-IR detection. [arXiv:2003.02437](https://arxiv.org/abs/2003.02437) | optional domain |
| `antiuav410` | Anti-UAV410 thermal UAV tracking benchmark | small-target regime |
| `cst-antiuav` | CST Anti-UAV, ICCV Workshops 2025 | tiny-target regime |

---

## Format

Add `docs/references.bib` as entries are confirmed, and cite by the keys above throughout
`docs/` and in code comments — keys, not prose descriptions, so a reader can check.
