
# References

**This file is load-bearing.** Every statement in this repository about what previous work
found must point at an entry here. A claim without a reference beside it is a claim this
repository does not make.

Prior-work findings used in a public release must be traceable to public sources.
A finding from a private requirements document is not a citable baseline merely
because that document exists in a local worktree; exclude it pending provenance review.

> **Verification status.** Entries marked *(verify)* were collected from abstracts and
> summaries and must be checked against the PDF before any number is quoted in a paper or a
> figure caption. Do this once, at the start of Phase 1, and strike the marker.

---

## The state of the question

The premise worth stating clearly: **reported downstream effects vary with the data, task and
protocol.** Recent work also proposes task-aware quality metrics, so a broad claim that nobody
predicts detection utility is no longer defensible. This repository tests the narrower question
of whether sensor-aware radiometric statistics add held-out predictive value for infrared data.

| Position | Source | What it measured |
|---|---|---|
| Mixing synthetic with real **helps**, with a non-monotonic optimum | `vanherle2022` | fixed-total mixing, industrial objects, Mask R-CNN |
| Synthetic **trails** real in-domain but **beats** it out-of-domain | `mayr2024` | thermal IR segmentation, two datasets |
| Synthetic-trained detectors differ from real-trained ones **layer by layer**, head most | `sundstrom2023` | YOLOv3, CKA analysis |
| Generated IR augmentation **helps** small-target detection | `kim2022`, `hybrid2024` | infrared small target detection |
| Detection-specific quality metrics can correlate with downstream AP | `sdqm2025`, `ccdm2026` | synthetic detection datasets and composition-stratified evaluation |
| Task-oriented IR generation can optimize downstream perception | `taskir2026` | IR detection and segmentation |

The remaining testable gap is narrower: whether a radiometric/sensor-aware diagnostic improves
over perceptual and detection-specific metrics when the generator and infrared domain are held
out, while initialisation, budget mode and data leakage are controlled.

---

## Protocol source for E1

E1 transfers a **published, citable** controlled-mixing design to infrared detection. Mirror
the decision into `configs/experiment/e1_reproduce.yaml:protocol_source`.

**Anchor: `vanherle2022`.** It is a public controlled study with a fixed total image count,
systematic ratio sweep and evaluation on real data. It used DIMO, Mask R-CNN/ResNet-101 and its
own training recipe. E1 uses FLIR and YOLOX-s, so it is a protocol transfer, not a reproduction.

| Field | Value |
|---|---|
| Citation | `vanherle2022` |
| Venue / year | BMVC 2022 (arXiv:2211.16066) |
| Dataset and split | DIMO (industrial metal objects); E1 transfers the protocol to FLIR ADAS v2 thermal |
| Detector and initialisation | Mask R-CNN + ResNet-101 with transfer learning; E1 uses YOLOX-s from scratch |
| Training | 100 epochs in the source experiment; E1 uses its explicitly recorded IR recipe |
| Reported trend | mixing outperformed real-only at several ratios, with an interior best ratio |
| Interpretation rule | report E1's curve as an IR result; do not require its magnitude or shape to match DIMO |

**Thermal precedent: `mayr2024`.** Report alongside E1. It is the strongest published evidence
that the sign of the effect depends on the evaluation setting rather than on the data: in-domain
the synthetic-trained model trails the real-trained one, and on an unseen dataset it exceeds it
*(verify)*. This is the published form of the coverage-gap hypothesis that F2 and E5 test.

An exact reproduction, if desired, must use the authors' public DIMO code/data and model recipe
as a separate experiment. E1 first qualifies this repository's real-only IR baseline, then
reports a three-arm controlled curve under stated conditions.

---

## Synthetic and generated data for detection

| Key | Work | Used for |
|---|---|---|
| `vanherle2022` | Analysis of Training Object Detection Models with Synthetic Data. BMVC 2022. [arXiv:2211.16066](https://arxiv.org/abs/2211.16066) · [code](https://github.com/EDM-Research/DIMO_ObjectDetection) | **E1 protocol source.** Fixed-total mixing design |
| `sundstrom2023` | Object Detector Differences when using Synthetic and Real Training Data. SN Computer Science 2023. [arXiv:2312.00694](https://arxiv.org/abs/2312.00694) | layer-wise account of *where* synthetic training diverges; motivates the stratified analysis in Phase 3 |
| `diversity2023` | Diversity and Diffusion: Observations on Synthetic Image Distributions with Stable Diffusion. [arXiv:2311.00056](https://arxiv.org/abs/2311.00056) | diversity collapse as a candidate mechanism; relates to `RFSSelector(mode="coverage")` |
| `sdqm2025` | SDQM: Synthetic Data Quality Metric for Object Detection Dataset Evaluation. [arXiv:2510.06596](https://arxiv.org/abs/2510.06596) | direct task-aware baseline for RFS |
| `ccdm2026` | Training-Free Metrics for Synthetic Object Detection Data: A Proxy for Detector Performance (Conditional-Composition Domain Match, CCDM). [arXiv:2606.19817](https://arxiv.org/abs/2606.19817) | direct training-free, composition-stratified baseline for RFS |
| `marginal-ap2025` | Online Data Curation for Object Detection via Marginal Contributions to Dataset-level AP. [arXiv:2511.14197](https://arxiv.org/abs/2511.14197) | `MarginalAPSelector` baseline |

## Thermal / infrared specific

| Key | Work | Used for |
|---|---|---|
| `mayr2024` | Narrowing the Synthetic-to-Real Gap for Thermal Infrared Semantic Image Segmentation Using Diffusion-based Conditional Image Synthesis. CVPR Workshops (PBVS). [openaccess](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Mayr_Narrowing_the_Synthetic-to-Real_Gap_for_Thermal_Infrared_Semantic_Image_Segmentation_CVPRW_2024_paper.pdf) | **thermal precedent for a domain-dependent sign**; the published form of the F2 / E5 hypothesis |
| `kim2022` | GAN-Based Synthetic Data Augmentation for Infrared Small Target Detection. IEEE TGRS 2022. [IEEE](https://ieeexplore.ieee.org/document/9786867/) | positive result in the small-target regime; E5 counterpoint |
| `hybrid2024` | Infrared Small Target Detection Improvement via Hybrid Data Augmentation Using Diffusion Models and GAN. IEEE 2024 | positive result; generator comparison |
| `irsurvey2024` | A Comprehensive Survey on Synthetic Infrared Image Synthesis. [arXiv:2408.06868](https://arxiv.org/abs/2408.06868) | IR radiometry background; simulation-tool landscape for the `synthetic_baseline` arm |
| `maritime2023` | Thermal-Infrared Remote Target Detection with 3D Game-Based Synthetic Data Augmentation. [arXiv:2310.20412](https://arxiv.org/abs/2310.20412) | prior art for rendered-synthetic IR training data |
| `taskir2026` | Taming Generative Diffusion Model for Task-Oriented Infrared Imaging. CVPR 2026. [openaccess](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Taming_Generative_Diffusion_Model_for_Task-Oriented_Infrared_Imaging_CVPR_2026_paper.html) · [code](https://github.com/csmty/InfraredIR) | direct IR generation precedent with downstream detection and segmentation |

## Generation

| Key | Work | Used for |
|---|---|---|
| `diffv2ir2025` | DiffV2IR: visible-to-infrared diffusion with vision-language understanding. [arXiv:2503.19012](https://arxiv.org/abs/2503.19012) | generator adapter, `configs/generator/diffv2ir.yaml` |
| `pid2026` | PID: Physics-Informed Diffusion Model for Infrared Image Generation. Pattern Recognition. [arXiv:2407.09299](https://arxiv.org/abs/2407.09299) · [code](https://github.com/fangyuanmao/PID) | temperature / emissivity / reflected-radiance conditioning, `configs/generator/pid_tev.yaml`; its reported evaluation uses image-quality metrics rather than downstream detection |
| `physirsplat2026` | PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting. CVPR 2026. [page](https://cvpr.thecvf.com/virtual/2026/poster/37591) | N3 background, `src/aero_ir/scene3d/plan.md`. Reconstruction quality, not detection utility |

## Detection and data

| Key | Work | Used for |
|---|---|---|
| `yolox` | YOLOX. [official code](https://github.com/Megvii-BaseDetection/YOLOX) | detector; pin a revision and record every recipe override |
| `superfusion2022` | SuperFusion: A Versatile Image Registration and Fusion Network with Semantic Awareness. [paper](https://www.ieee-jas.net/article/doi/10.1109/JAS.2022.106082) · [official code](https://github.com/Linfeng-Tang/SuperFusion) | frozen image-conditioned RGB/IR dense-registration baseline; vendored matcher code retains MIT attribution |
| `rift2019` | RIFT: Multi-modal Image Matching Based on Radiation-variation Insensitive Feature Transform. IEEE TIP. [paper](https://doi.org/10.1109/TIP.2019.2959244) | motivates structure/phase-sensitive evidence instead of raw cross-modal intensity agreement |
| `nemar2020` | Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation. CVPR. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Arar_Unsupervised_Multi-Modal_Image_Registration_via_Geometry_Preserving_Image-to-Image_Translation_CVPR_2020_paper.html) · [official code](https://github.com/moabarar/nemar) | motivates geometry preservation and separating modality appearance from spatial alignment |
| `xoftr2024` | XoFTR: Cross-modal Feature Matching Transformer. CVPR Workshops. [paper](https://openaccess.thecvf.com/content/CVPR2024W/IMW/html/Tuzcuoglu_XoFTR_Cross-modal_Feature_Matching_Transformer_CVPRW_2024_paper.html) · [official code](https://github.com/OnderT/XoFTR) | visible/TIR-specific precedent for modality-aware matching and sub-pixel refinement |
| `roma2024` | RoMa: Robust Dense Feature Matching. CVPR. [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html) · [official code](https://github.com/Parskatt/RoMa) | robust coarse-to-fine dense correspondence and explicit match confidence precedent |
| `minima2025` | MINIMA: Modality Invariant Image Matching. CVPR 2025. [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html) · [official code](https://github.com/LSXI7/MINIMA) | synthetic cross-modal training motivates the fixed-protocol XoFTR checkpoint comparison; local measurements and limitations in `registration_minima.md` |
| `c2rf2025` | C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning. IJCV. [official code](https://github.com/QinglongYan-hub/C2RF) | cross-modal commonality and contrastive structural evidence for registration/fusion |
| `flir-adas-v2` | Teledyne FLIR ADAS Thermal Dataset v2. [official access page](https://oem.flir.com/en-gb/solutions/automotive/adas-dataset-form/) | IR protocol-transfer domain; visible/thermal pairing requires an audited manifest |
| `llvip` | LLVIP visible-infrared pairs, aligned by the authors using manually selected point pairs, projective warping and cropping. [arXiv:2108.10831, §3](https://arxiv.org/abs/2108.10831) | optional control domain, not independent Anti-UAV correspondence evidence; residual accuracy and licensing need auditing |
| `uav-tirvis` | UAV-TIRVis: A Benchmark Dataset for Thermal–Visible Image Registration from Aerial Platforms. [paper](https://doi.org/10.3390/jimaging11120432) · [official data](https://gitlab.upb.ro/etti/dcae-public/arh/research/uav-tirvis) | external landmark diagnostic only; ground-scene domain, reference uncertainty and redistribution limitations recorded in `registration_external_landmarks.md` |
| `dronevehicle` | DroneVehicle aerial RGB-IR detection. [arXiv:2003.02437](https://arxiv.org/abs/2003.02437) | optional domain |
| `antiuav300` | Anti-UAV300 RGB/IR tracking video pairs, not spatially registered pairs. [official project](https://github.com/ZhaoJ9014/Anti-UAV) · [benchmark paper, Fig. 1 and §III](https://arxiv.org/abs/2101.08466) | candidate small-target source; registration and independent correspondence remain unqualified, not approved paired-generator supervision |
| `antiuav410` | Anti-UAV410 IR-only tracking benchmark. [project](https://github.com/HwangBo94/Anti-UAV410) | sequence-disjoint external thermal evaluation |
| `cst-antiuav` | CST Anti-UAV, ICCV Workshops 2025 | tiny-target regime |

---

## Format

Add `docs/references.bib` as entries are confirmed, and cite by the keys above throughout
`docs/` and in code comments — keys, not prose descriptions, so a reader can check.
