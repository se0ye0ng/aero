# AERO — When Does Generated Infrared Data Help Detection?

**A radiometric-consistency study of generative training data for infrared target detection.**

> Published results on whether synthetic and generated imagery helps a detector
> **disagree with each other**, and the disagreement is unexplained. This repository tests
> whether it is explained by the training protocol and by the fidelity criterion used to
> judge the data.
>
> Every claim about prior results is anchored to a citation in
> [`docs/references.md`](docs/references.md). Nothing in this repository derives from
> non-public material — see [Provenance](#provenance).

**Central hypothesis.** Perceptual similarity (FID / LPIPS / SSIM) does not predict whether
generated IR data helps a detector. **Radiometric consistency does.**

---

## Contents

- [Why this repository exists](#why-this-repository-exists)
- [Phases](#phases) — the implementation plan, phase by phase
- [Moving this repository to a GPU machine](#moving-this-repository-to-a-gpu-machine)
- [Compute budget](#compute-budget)
- [The pipeline](#the-pipeline)
- [What the controlled protocol fixes](#what-the-controlled-protocol-fixes)
- [Repository layout](#repository-layout)
- [Mapping to industry requirements](#mapping-to-industry-requirements)
- [Data](#data)
- [Reproducibility](#reproducibility)

---

## Why this repository exists

Real infrared imagery is expensive to acquire: it is bounded by physics (range, aspect,
atmosphere) and by time (flight hours, range slots). Every operational IR detection program
therefore falls back on simulated or generated imagery, and every one of them runs into the
same question — *can this data be trusted for training, and under what conditions?*

The published answers do not agree. Controlled mixing studies report that adding synthetic
data *helps*, with a non-monotonic optimum. Thermal-specific work reports that synthetic
training data sits slightly *below* real data in-domain and *above* it on an unseen domain —
the sign of the effect changes with the evaluation setting alone. Practitioner reports of
generated IR degrading detection are common and, in the cases that are documented, share a
set of experimental conditions that would bias the result negative.

Nobody has controlled those conditions and measured which one is responsible. That is what
this repository does. See [`docs/references.md`](docs/references.md) for the specific
positions and what each one measured.

---

## Phases

Each phase states what to implement, how to run it, and **the condition that must hold before
moving on**. The exit criteria are not formalities: Phase 1 in particular gates everything
after it, because a result that cannot be reproduced cannot be reinterpreted.

| Phase | Goal | State | Runs |
|---|---|---|---|
| [0](#phase-0--scaffold-and-instrumentation) | Scaffold, sensor chain, RFS diagnostic | **done** | 0 |
| [1](#phase-1--data-layer-and-the-e1-reproduction) | Data layer, detector, **three-arm reproduction** | next | 42 |
| [2](#phase-2--the-controlled-experiment-f1-f3) | Pretraining and budget ablations — **the sign-flip test** | | 108 |
| [3](#phase-3--label-audit-and-failure-mode-decomposition-f4-f5) | Label audit, stratified analysis | | 0 |
| [4](#phase-4--rfs-predictive-power-n1) | Curation policies, **RFS vs FID predictive power** | | 30 |
| [5](#phase-5--small-target-regime-and-sensor-matching) | Small-target regime with a real coverage gap | | 36 |
| [6](#phase-6--sensor-in-the-loop-generation-n2) | Differentiable sensor in the generation loop | | ~20 |
| [7](#phase-7--deployment-track) | ONNX / INT8 / latency-vs-mAP | | 0 |
| [8](#phase-8--radiometrically-consistent-3d-generation-n3) | 3D multi-view IR generation | | TBD |
| [9](#phase-9--outputs) | Preprint, figures, cards | | 0 |
| [E6](#optional--e6-capacity) | *optional* — does capacity change the effect? | | 36 |

**The three arms.** Every mixing experiment compares **real**, **real + simulated**, and
**real + generated**. The middle arm is `synthetic_baseline` — a deterministic, unlearned
pseudo-IR renderer (`src/aero_ir/generate/synthetic_baseline.py`). Without it, a gain over
real-only could be a gain any crude simulation would also produce, and a loss could be a loss
any non-real imagery would produce. **The generative model is only interesting to the extent
it beats the free option.**

---

### Phase 0 — scaffold and instrumentation

**Done.** Present in this repository and covered by tests.

| Component | File | Status |
|---|---|---|
| Differentiable IR sensor chain (MTF, NETD, FPN/NUC, AGC + 8-bit) | `src/aero_ir/sensor/` | implemented, differentiability tested |
| Radiometric Fidelity Score R1–R8 | `src/aero_ir/rfs/stats.py` | implemented, behaviour tested |
| Distributional distances + real-set sampling floor | `src/aero_ir/rfs/distances.py`, `report.py` | implemented |
| Mixing budget semantics (`fixed_total` vs `additive`) | `src/aero_ir/data/mixing.py` | implemented, tested |
| Experiment grid E1–E5 as configuration | `configs/experiment/` | declared |
| Protocol, RFS spec, roadmap | `docs/` | written |

**Exit criterion.** `make smoke` passes on the target machine. This needs neither GPU nor
data, so it is the first thing to run after transfer.

---

### Phase 1 — data layer and the E1 reproduction

**The gate.** If the reported negative result cannot be reproduced under its original
conditions, no later experiment is interpretable — a changed outcome could be the change in
conditions or could be a difference in the setup. Do not proceed past this phase on a partial
reproduction.

**Implement**

| File | What |
|---|---|
| `src/aero_ir/data/registry.py` | FLIR ADAS v2 loader, **thermal and the aligned visible split** — the visible images are the input the simulated and generated arms both render from. Contract: images as float (H, W) in a consistent intensity unit; boxes `(x, y, w, h)` COCO; metadata carrying the key named by `data.held_out_scenario.key` |
| `src/aero_ir/detect/yolox_adapter.py` | `fit()` / `predict()` around the upstream trainer. Augmentation is fixed across arms by protocol |
| `src/aero_ir/detect/evaluate.py` | `coco_metrics()`; `delta_ap()` is already implemented |
| `src/aero_ir/cli.py` | dispatch for `run` |
| `scripts/run_grid.py` | replace the `pass` with a launcher call |

**Run**

```bash
export AERO_DATA_ROOT=/path/to/data
bash scripts/download_flir.sh          # prints access route, verifies layout
python scripts/run_grid.py e1_reproduce --dry-run
make e1
```

**Exit criterion.** The trend reported by the published baseline named in
[`docs/references.md`](docs/references.md) is recovered within its stated tolerance, across
3 seeds. Fill in that file before starting: **E1 reproduces a citable public result, not an
impression.** Record the reproduced curve — it is Figure 1's baseline and the thing every
later result is measured against.

**Compute.** 2 generators x 7 ratios x 3 seeds = **42 runs**, YOLOX-s from scratch
(300 epochs — the expensive phase, by construction, since scratch training is the condition
being tested).

**Also delivers a result on its own.** `synthetic_baseline` versus `diffv2ir` at matched
ratio and budget answers a question the field mostly assumes: *does the generative step beat
a crude simulator?* If it does not, that is a finding, and it reframes everything after it.

---

### Phase 2 — the controlled experiment (F1, F3)

**Implement**

| File | What |
|---|---|
| `src/aero_ir/utils/manifest.py` | wire `RunManifest` into every run; dataset checksums |
| `scripts/verify_run.py` | re-execute from a manifest, diff metrics within tolerance |
| W&B logging | `tracking.wandb` in `configs/config.yaml` |

**Run**

```bash
make e2     # train: [scratch, pretrained] x gen_ratio x seed   -> the sign-flip test
make e3     # budget_mode: [fixed_total, additive] x gen_ratio x seed
```

**Exit criterion.** A decision, with a bootstrap CI over seeds, on:

1. **Does the sign of `dAP` change with initialisation alone?** If yes, the headline result is
   in hand: the reported negative finding is an artifact of scratch training.
2. **Does `fixed_total` differ from `additive`?** These answer different questions —
   *substituting* generated for real data versus *adding* it. Reporting them separately is
   what separates "generated data harms" from "having less real data harms".

A null result here is still a result, and a publishable one: it would establish that the
negative finding is robust to the two conditions most likely to explain it, which no
published work currently shows. Do not treat a null as a failed phase.

**Compute.** E2: 2 inits x 4 ratios x 3 seeds = 24 runs. E3: 2 generators x 2 budget modes
x 7 ratios x 3 seeds = 84 runs. **108 runs.** E3 uses pretrained initialisation (100 epochs),
so it is roughly a third the cost per run of E1.

---

### Phase 3 — label audit and failure-mode decomposition (F4, F5)

No new training. This phase re-analyses the runs Phase 2 produced, and it is where the
analysis stops being a single number.

**Implement**

| File | What |
|---|---|
| `src/aero_ir/data/labels.py` | `audit()` — IoU drift, centroid shift, area ratio, contrast floor. Reuses `rfs.stats.target_snr` so the audit and the fidelity diagnostic share one definition of contrast |
| `src/aero_ir/analysis/failure_modes.py` | `stratified_ap()` over target pixel area, contrast quartile, polarity, clutter |

**Exit criterion.** Every reported result carries its label-exclusion rate, and `dAP` is
decomposed by stratum. Concretely: the answer to *"where did the effect actually land?"* is a
table, not a guess. If the effect is confined to the tiny-target bucket, that is the finding
and it reshapes the paper.

**Compute.** None beyond inference over existing checkpoints.

---

### Phase 4 — RFS predictive power (N1)

**The core contribution.** Everything before this establishes that the effect is real and
measurable; this phase asks whether it can be *predicted* — and therefore controlled.

**Implement**

| File | What |
|---|---|
| `src/aero_ir/curate/selectors.py` | `RFSSelector` (per-sample RFS, `closest` and `coverage` modes), `PerceptualSelector` (per-sample distance to the real feature centroid), `MarginalAPSelector` (gradient-alignment proxy) |
| `src/aero_ir/analysis/predictive_power.py` | regress `dAP` on FID / LPIPS / SSIM / RFS scalar / RFS vector; Spearman and cross-validated R² |

**Run**

```bash
make e4     # curation: [none, random, fid_topk, rfs_topk, marginal_ap] x ratio x seed
```

**Exit criterion.** A single table: predictive power of each fidelity measure for `dAP`.

- **H1** holds if FID and LPIPS have low predictive power.
- **H2** holds if the RFS scalar and vector beat them.
- **H4** holds if `rfs_topk` beats `random` and `fid_topk` at an equal budget.

If RFS loses to FID, say so plainly and report it. An honest negative on the central
hypothesis is worth more than a hedged positive, and it still answers a question nobody has
answered.

**Compute.** 5 policies x 2 ratios x 3 seeds = **30 runs**, pretrained.

---

### Phase 5 — small-target regime and sensor matching

The regime operational IR detection actually lives in, and the first configuration where
generated data has a genuine coverage gap to fill rather than existing scenes to restyle.

**Implement**

| File | What |
|---|---|
| `src/aero_ir/data/registry.py` | Anti-UAV410 loader; `target_pixel_area_bin` metadata for the held-out slice |
| `src/aero_ir/sensor/fit.py` | `fit_sensor_params()` — NETD from the noise PSD floor (R5), MTF cutoff from the spectrum roll-off (R4), column FPN from the variance ratio (R8), AGC clip points from the histogram (R7) |

**Run**

```bash
bash scripts/download_antiuav.sh
make e5     # gen_ratio x sensor: [none, eo_ir_default, matched] x seed
```

**Exit criterion.** Is `dAP > 0` in the held-out scenario slice, and does the fitted sensor
model beat both the generic profile and no sensor model at all? This is the cell that decides
whether the whole thesis generalises past the driving/urban domain.

**Compute.** 4 x 3 x 3 = **36 runs**.

---

### Phase 6 — sensor-in-the-loop generation (N2)

Everything up to here applies the sensor model as post-processing. This phase puts it *inside*
the loop: a detection loss back-propagated through the sensor chain into the generator, so
generation is optimised for the detector's decision statistics rather than for perceptual
realism. The sensor stages are already `torch.nn.Module`s with a straight-through quantiser
for exactly this reason, and `tests/test_sensor.py::test_chain_is_differentiable_end_to_end`
guards the gradient path.

**Implement**: `src/aero_ir/generate/tev.py`, generator fine-tuning with the sensor chain and
a frozen detector in the graph.

**Open questions to resolve here, not before**: which sensor parameters are identifiable from
imagery alone, and whether back-propagating through AGC needs the soft surrogate
(`AGCQuantise(soft_quantise=True)`) rather than the straight-through estimator.

**Exit criterion.** Generated data from the in-loop generator beats post-processed generated
data on `dAP` at an equal budget.

**Compute.** Generator fine-tuning dominates. Budget ~20 detector runs plus generator training.

---

### Phase 7 — deployment track

Small in effort, and it is the difference between a research repository and one that reads as
deployable. Can be done any time after Phase 2 has a trained checkpoint.

**Implement**: `src/aero_ir/deploy/export.py` — ONNX export, INT8 post-training quantisation
calibrated on the real training split, latency percentiles on a fixed device profile.

```bash
make deploy-bench
```

**Exit criterion.** One latency-versus-mAP curve. The curve is the deliverable, not a single
quantised number: what matters operationally is *where* accuracy starts to fall off.

**Compute.** Hours on one GPU.

---

### Phase 8 — radiometrically consistent 3D generation (N3)

The only track that addresses the original constraint — imagery that *cannot be acquired* —
rather than restyling imagery that already exists. Scaffolded in `src/aero_ir/scene3d/`;
the plan is in `src/aero_ir/scene3d/plan.md`.

Physically consistent thermal 3D reconstruction has been published and is evaluated on
reconstruction quality. The open question is whether such views work as **training data**,
which is what Phases 1–5 build the instrument to measure. The differentiation is the
evaluation axis and the small-target regime, not the renderer.

**Do not start this before Phase 5 reports.** Without a working `dAP` measurement in the
held-out-scenario setting, there is nothing to evaluate the rendered views against, and the
phase collapses into a reconstruction-quality exercise that has already been done.

---

### Optional — E6 capacity

Does the effect of generated data depend on detector capacity? Every published result on this
question is single-model, so nobody knows. `make e6` sweeps YOLOX tiny/s/m/l. Run it after E4
reports; if the sign of `dAP` is capacity-dependent, that constrains how any of these
conclusions may be stated.

**Compute.** 4 sizes x 3 ratios x 3 seeds = 36 runs.

---

### Phase 9 — outputs

- `scripts/make_report.py` regenerates all five figures and three tables from the runs table.
  Figures are derived, never hand-edited.
- Preprint; model card, data card, environment lock.
- Every claim traceable to a run id in `experiments/`.

---

## Moving this repository to a GPU machine

Nothing here is bound to the machine it was written on. `data/` and `experiments/` are
git-ignored, so a clone carries code and configuration only.

### 1. Transfer

```bash
# preferred - carries history
git clone <remote> aero-ir && cd aero-ir

# or, direct copy; exclude generated state explicitly
rsync -av --exclude '.git' --exclude 'data' --exclude 'experiments' \
      --exclude '__pycache__' --exclude '.pytest_cache' \
      aero-ir/ user@gpu-host:~/aero-ir/
```

### 2. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# install torch first, matched to the host CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -e ".[torch,track,detect,deploy,dev]"
pre-commit install
```

`torch` is an optional extra rather than a hard dependency precisely so the analysis and RFS
code stay installable on a machine without CUDA.

### 3. Verify the transfer — before touching data or GPUs

```bash
make smoke
```

Runs the full test suite plus an RFS end-to-end check on synthetic scenes. On a correct
transfer a radiometrically degraded set scores far from the real set and the report names
the components that failed. If this passes, the code arrived intact.

### 4. Data

```bash
export AERO_DATA_ROOT=/mnt/data/aero        # put this in your shell profile
bash scripts/download_flir.sh               # prints the access route, verifies layout
```

Datasets are gated and are never fetched automatically. The scripts verify layout and record
checksums after manual placement.

### 5. Multi-GPU

The workload is many small runs, not one large one. **One run per GPU beats data-parallel
across GPUs here** — a 153-run grid finishes far sooner with 8 independent runs in flight
than with 8-way DDP on one run at a time, and it keeps seeds cleanly separated.

```bash
# shard a grid across 8 GPUs
python scripts/run_grid.py e3_mixing_ratio --dry-run > /tmp/e3.txt
awk 'NR % 8 == 0' /tmp/e3.txt | while read -r cmd; do CUDA_VISIBLE_DEVICES=0 $cmd; done &
# ... one such loop per device, or use a job launcher / hydra multirun
```

Set `tracking.wandb.mode=offline` on a host without outbound network and sync later with
`wandb sync`.

---

## Compute budget

Order-of-magnitude planning figures for YOLOX-s on a single 48 GB card. **Measure one run
before trusting the totals** — the numbers below exist to size the grid, not to promise a
schedule.

| Phase | Runs | Init | Epochs | Rough per-run | Wall-clock on 8 GPUs |
|---|---|---|---|---|---|
| 1 (E1) | 42 | scratch | 300 | ~2 h | ~11 h |
| 2 (E2) | 24 | mixed | 300 / 100 | ~1.3 h | ~4 h |
| 2 (E3) | 84 | pretrained | 100 | ~0.7 h | ~8 h |
| 4 (E4) | 30 | pretrained | 100 | ~0.7 h | ~3 h |
| 5 (E5) | 36 | pretrained | 100 | ~0.7 h | ~3 h |
| **core total** | **216** | | | | **~29 h** |
| E6 (optional) | 36 | pretrained | 100 | ~0.4–1.5 h | ~4 h |

Phases 3, 7 and 9 add no training. Phase 6 is dominated by generator fine-tuning and is
budgeted separately once Phase 5 reports.

**Cut if the budget is tight**, in this order: E6 entirely; then E3's `fixed_total` half
(keep `additive`, which is the operationally meaningful mode); then E1's ratios 0.6 and 0.8.
Never cut seeds — a two-seed result cannot support a claim about a few AP points.

Storage: budget roughly 200 GB for datasets, generated sets and checkpoints combined.

---

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

| | Common property | Why it invalidates the conclusion | Handled by | Phase |
|---|---|---|---|---|
| F1 | Detector trained from random init | Un-pretrained detectors are maximally sensitive to distribution shift; production training never starts from scratch | `train.pretrained` as an explicit factor | 2 |
| F2 | Generated data derived from the *same scenes* as the real data | Adds noise without adding coverage — the entire point of synthetic data is untested | `data.coverage_split` | 5 |
| F3 | Mixing ratio confounded with total dataset size | A ratio increase may mean less real data *or* more total data; the two imply opposite conclusions | `mixing.budget_mode` | 2 |
| F4 | Labels transferred across the generation step, unaudited | If generation deforms object boundaries, label noise enters silently | `data.labels.audit` | 3 |
| F5 | A single aggregate AP number | Cannot tell *which* failure mode moved, so the analysis terminates in a question mark | `analysis.failure_modes` | 3 |

## Repository layout

```
docs/                     problem statement, experiment protocol, RFS spec, datasets, roadmap
configs/                  Hydra tree - every experiment declared here, none in scripts
  experiment/             E1-E5, each with its sweep and success criterion
src/aero_ir/
  sensor/                 differentiable IR sensor chain          [Phase 0 done]
  rfs/                    radiometric fidelity diagnostic R1-R8   [Phase 0 done]
  data/                   loaders, mixing budget, label audit     [mixing done]
  generate/               generator adapters behind one Protocol
  curate/                 selection policies
  detect/                 detector adapter, COCO metrics, dAP     [dAP done]
  analysis/               failure modes, predictive power
  deploy/                 ONNX / INT8 / latency
  scene3d/                N3 scaffold + plan.md
scripts/                  dataset access, grid expansion, report, run verification
tests/                    15 tests; no GPU or dataset required
```

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

## Data

Public datasets only. No proprietary imagery, labels, or specifications are used or referenced.

| Dataset | Role | Phase |
|---|---|---|
| Teledyne FLIR ADAS Thermal v2 | reproduction anchor — the driving/urban regime | 1 |
| LLVIP | aligned low-light visible-IR pairs | optional |
| Anti-UAV410 / CST Anti-UAV | small, low-contrast target regime | 5 |
| DroneVehicle | aerial viewpoint, oriented boxes | optional |

See [`docs/datasets.md`](docs/datasets.md) for licences and access.

## Provenance

This repository is built from public sources only, and is written so that fact is checkable
rather than merely asserted.

- **Prior results** are cited in [`docs/references.md`](docs/references.md). A statement about
  what previous work found is not made in this repository without a reference beside it.
- **Data** is public and gated only by the providers' own request forms. See
  [`docs/datasets.md`](docs/datasets.md).
- **Baselines** — YOLOX and the generator checkpoints — are public releases. Training
  hyperparameters follow the upstream YOLOX defaults (300 epochs, batch 64) so that the
  reproduction arm is a documented public recipe rather than a borrowed configuration.
- **No proprietary material** of any kind: no imagery, labels, sensor specifications,
  requirement documents, internal results, or organisation names. This is enforced by
  `.gitignore` and stated as the first rule in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Reproducibility

Every run records: config hash, git SHA, dataset checksum, seed, environment lock, and the
full metric set. `make verify RUN=<id>` re-executes a run and diffs its metrics. A run that
does not reproduce within tolerance is marked and excluded from reported aggregates.

The claims here concern effect signs of a few AP points. Without seed control and a reported
spread, an effect that size is indistinguishable from run-to-run noise — which is why every
cell in every grid is three seeds and every headline number carries a bootstrap interval.

## Licence

MIT. See [`LICENSE`](LICENSE).
