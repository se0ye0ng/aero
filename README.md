# AERO — When Does Generated Infrared Data Help Detection?

**A radiometric-consistency study of generative training data for infrared target detection.**

> Generated infrared imagery is widely reported to *degrade* detector performance.
> This repository tests whether that conclusion is a property of generated data
> or an artifact of the training protocol and of the fidelity criterion used to judge it.

**Central hypothesis.** Perceptual similarity (FID / LPIPS / SSIM) does not predict whether
generated IR data helps a detector. **Radiometric consistency does.**

---

## Why this repository exists

Real infrared imagery is expensive to acquire: it is bounded by physics (range, aspect,
atmosphere) and by time (flight hours, range slots). Every operational IR detection program
therefore falls back on simulated or generated imagery, and every one of them runs into the
same question — *can this data be trusted for training, and under what conditions?*

Published attempts to answer it report a negative result: mixing generated IR into the
training set lowers mAP, and the effect worsens with the mixing ratio. This repository
reproduces that result, then shows which experimental conditions it depends on.

## Status

| Stage | State |
|---|---|
| Repository scaffold, experiment protocol, configs | done |
| E1 reproduction of the negative result (FLIR) | in progress |
| E2 pretraining ablation — the sign-flip test | in progress |
| E3 mixing ratio x budget, E4 curation, E5 small-target regime | planned |
| N2 sensor-in-the-loop generation | planned |
| N3 radiometrically consistent 3D multi-view generation | planned |

## The pipeline

```
S1  Scenario definition          acquirable vs non-acquirable scenarios; coverage gap made explicit
S2  Physics-conditioned          T / epsilon / V decomposition + differentiable IR sensor model
    generation                   (NETD, MTF, AGC + 8-bit quantisation, FPN / NUC residual)
S3  Radiometric Fidelity Score   diagnostic vector, not a scalar:
    (RFS)                        target-background dT, thermal polarity, radial spectrum,
                                 noise PSD, per-class target SNR, target pixel-area
S4  Utility-aware curation       select / weight generated samples under a fixed budget
S5  Controlled training          (pretrained on/off) x (fixed-N / additive) x (mix ratio)
                                 x (domain) x 3 seeds, with a label-transfer audit
S6  Diagnostics + deployment     failure-mode decomposition; ONNX -> TensorRT INT8;
                                 latency vs mAP trade-off
```

Full design rationale: [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## What the controlled protocol fixes

Reported negative results on generated IR share five design properties that this protocol
removes. Each one is a factor in the run grid rather than an assumption.

| | Common property | Why it invalidates the conclusion | Handled by |
|---|---|---|---|
| F1 | Detector trained from random init | Un-pretrained detectors are maximally sensitive to distribution shift; production training never starts from scratch | `train.pretrained` as an explicit factor (E2) |
| F2 | Generated data derived from the *same scenes* as the real data | Adds noise without adding coverage — the entire point of synthetic data is untested | `data.coverage_split` — generation restricted to held-out scenarios (E5) |
| F3 | Mixing ratio confounded with total dataset size | A ratio increase may mean less real data *or* more total data; the two imply opposite conclusions | `mixing.budget_mode` in {`fixed_total`, `additive`} (E3) |
| F4 | Labels transferred across the generation step, unaudited | If generation deforms object boundaries, label noise enters silently | `data.labels.audit` — IoU drift and boundary-shift report |
| F5 | A single aggregate AP number | Cannot tell *which* failure mode moved, so the analysis terminates in a question mark | `analysis.failure_modes` — stratified by target size, contrast, polarity, clutter |

## Mapping to industry requirements

Public job descriptions in defence EO/IR R&D repeatedly ask for the same capabilities.
This table exists so a reviewer can find the corresponding code in one step.

| Required capability | Module | Entry point |
|---|---|---|
| Generation of synthetic EO/IR imagery for training | `aero_ir.generate` | `configs/generator/` |
| Construction and curation of a training database | `aero_ir.data`, `aero_ir.curate` | `configs/curation/` |
| Automatic target detection / recognition models | `aero_ir.detect` | `configs/detector/` |
| Physics-based sensor and scene modelling (M and S) | `aero_ir.sensor`, `aero_ir.scene3d` | `configs/sensor/` |
| On-device optimisation and profiling | `aero_ir.deploy` | `make deploy-bench` |
| Reproducible experiment control and configuration management | Hydra + DVC + W and B | `configs/`, `dvc.yaml` |

## Quickstart

```bash
git clone <this-repo> && cd aero-ir
make setup                 # editable install + pre-commit hooks
make data-flir             # dataset fetch instructions and integrity check
make e1                    # E1: reproduce the reported negative result
make e2                    # E2: pretraining ablation - the sign-flip test
make report                # regenerate figures and tables under experiments/
```

All experiments are declared in `configs/experiment/`. Nothing is hard-coded in a script.

## Data

Public datasets only. No proprietary imagery, labels, or specifications are used or referenced.

| Dataset | Role |
|---|---|
| Teledyne FLIR ADAS Thermal v2 | reproduction anchor — the driving/urban regime |
| LLVIP | aligned low-light visible-IR pairs |
| Anti-UAV410 / CST Anti-UAV | small, low-contrast target regime |
| DroneVehicle | aerial viewpoint, oriented boxes |

See `docs/datasets.md` for licences and access.

## Reproducibility

Every run records: config hash, git SHA, dataset checksum, seed, environment lock, and the
full metric set. `make verify RUN=<id>` re-executes a run and diffs its metrics.

## Licence

MIT. See `LICENSE`.
