# DiffV2IR integration: CPU-tested interfaces, pending real-model verification

## Current scope

`src/aero_ir/generate/diffv2ir_data.py` implements a train-only PyTorch dataset
with the conditioning keys expected by the official DiffV2IR model. It does not
fit registration, produce captions/masks, export registered frames, load diffusion
weights, train a diffusion model, or authorize generator training.
`DiffV2IRGenerator.generate()` now launches an isolated inference worker connected
to the pinned official sampling implementation. Its transport and geometry are
CPU-tested, but actual checkpoint loading and GPU inference are **not yet verified**.
Likewise,
`scripts/prepare_generator_training.py` is orchestration, not a diffusion trainer;
its plan or a string-valued gate must not be treated as qualification evidence.
The planner now recomputes the supported v4 audit's metrics and gates from its
retained bidirectional rows, verifies report/cache content hashes and shared
official-train identity, and requires explicit train-only cache metadata. A
manually edited `pass` string, even with a recomputed content hash, is rejected.
Unknown qualification protocols are not silently accepted. V4 engineering gates
do not authorize paired generation, and the raw registration cache is not a
qualified conditioning export. An actual scientific-qualification/export
authorizer remains to be implemented; this planner cannot currently launch a
legitimate generator training run. This is an explicit limitation, not a newly
invented pass rule.

CPU interface tests use synthetic fixtures, not generated IR experiment results.
The official source was cloned into git-ignored `experiments/external/DiffV2IR`.
No GPU experiment or checkpoint download has been performed for this component.
A separate candidate environment under `experiments/external/diffv2ir-runtime`
now passes the offline CPU upstream component probe described below. This does
not establish pretrained-checkpoint loading, real-data diffusion training or GPU compatibility.
The registration `.venv` and v7 training sources are unchanged.

A subsequent tiny **full training-path** CPU probe also passes after the narrow
Lightning1.9.5 hook adapter described below. It uses genuine upstream VAE, text
encoder and diffusion components, but only random small-model weights and
synthetic inputs. It is not real-data/pretrained-model training or qualification.

## Official implementation inspected

The [official repository](https://github.com/LidongWang-26/DiffV2IR) provides
training/inference entry points and checkpoint download links. Its existence is
not evidence of verified weights or a working GPU runtime. The inference bridge
requires revision `495947eaad0c2f3bb380ddf5f8820c2b1443e23f`, checks tracked source
integrity, and records hashes of the tracked files.

The upstream [dataset](https://github.com/LidongWang-26/DiffV2IR/blob/main/DiffV2IR_dataset.py)
partitions its ordered input list by fractions, generates stochastic captions in
sample loading, and applies shared random crops/flips. Our adapter instead requires
explicit official-train membership, frozen prompts and full-grid resize. These are
documented protocol differences, not an exact upstream recipe reproduction.
It returns `edited` as the IR target and `edit` with `c_concat1` (RGB), `c_concat2`
(rendered segmentation), and `c_crossattn` (prompt). The upstream
[model](https://github.com/LidongWang-26/DiffV2IR/blob/main/stable_diffusion/ldm/models/diffusion/ddpm_DiffV2IR.py)
uses an unmasked spatial diffusion objective; registration padding cannot simply
be supplied as a target. Equal image sizes alone do not demonstrate registration.

The published [training config](https://github.com/LidongWang-26/DiffV2IR/blob/main/configs/train.yaml)
contains author-machine paths. The [inference entry point](https://github.com/LidongWang-26/DiffV2IR/blob/main/infer.py)
also uses a fixed local BLIP checkpoint path. The [requirements file](https://github.com/LidongWang-26/DiffV2IR/blob/main/requirements.txt)
specifies an older Torch/Lightning environment and has single-equals version lines.
Do not install it into the working registration environment unchanged. Pin a
revision, review dependency and weight licenses, and test a separate environment
before running its model backend. The links above describe the inspected source;
the bridge pins the revision, but a tested dependency lock is still pending.

## Inference bridge and protocol differences

`diffv2ir_adapter.py` writes a frozen request and invokes
`scripts/sample_diffv2ir.py` using an explicitly configured `runtime_python`.
The worker loads the official model, its `CFGDenoiser`, and k-diffusion's Euler
ancestral sampler. This is executable integration code, not evidence that the
optional upstream runtime or checkpoint is compatible. Model errors propagate;
there is no identity-image or simulated-IR fallback.

Inputs must include local checkpoint bytes plus an expected SHA256, local CLIP
model/tokenizer assets, RGB images, matching rendered segmentation images,
frozen prompts, unique source IDs, and caption/segmentation specification hashes.
Prompt origin is declared by the caller; a hash alone does not verify that BLIP
actually produced it. All prompts and image bytes are retained with the request.
The worker runs offline, preserves scheduler `CUDA_VISIBLE_DEVICES`, and removes
`LD_LIBRARY_PATH` only from its own process environment. It must use a separately
prepared environment; do not install upstream dependencies into active v7 training.

Sampling differs from the upstream demonstration in explicit ways:

- Single-image batches are required by the official four-branch CFG layout.
- RGB and segmentation receive identical right/bottom edge padding to multiples
  of 64, followed by cropping **only the padding** after decoding. There is no
  centre crop or resize of source coordinates. Padding can still affect generation.
- Output quantization rounds to uint8 rather than truncating. These choices are
  not an exact reproduction of the upstream demonstration preprocessing.
- Each source gets a stable seed derived from the experiment seed and source ID.
  This does not guarantee bitwise determinism across GPU/runtime versions.
- Checkpoints load with `weights_only=True`, finite-tensor checks and strict
  state matching. An incompatible public checkpoint fails rather than silently
  initializing missing parameters; compatibility has not been established yet.

Each fresh output directory contains `request.json`, conditioning PNGs,
`worker.log`, generated PNGs and, only after successful checks, `result.json`.
The result records input/output identities, geometry, source hashes (including
the pinned editable taming-transformers dependency), local CLIP
hashes, resolved model config, runtime package versions and seeds. Partial failures
are preserved for inspection and require a new output directory for reruns.
Copied source labels remain **unverified candidates** requiring the F4 label
audit; equal image dimensions do not prove an object survived synthesis.
Checkpoint training-data exposure is not established by inference, and generated
RGB display intensities are not calibrated thermal radiance.

The editable taming-transformers source is checked both before model loading and
after sampling. A changed dependency prevents publication of `result.json`;
partial generated files remain for inspection, not as verified results.

No inference output authorizes registration or generator training. A pretrained
model's training datasets and permitted usage must be reviewed before using its
outputs in a held-out benchmark. The current bridge does not produce captions,
segmentations, a qualified registered export, or a diffusion training launcher.

## Export contract

The input manifest is an object with:

- `schema_version: 1`, `kind: aero_diffv2ir_train_conditioning`, `split: train`;
- `coordinate_domain: common_ir_observation_grid`;
- `official_train_sha256`: byte hash of the official `label_new/train.json`;
- `provenance`: SHA256 identities named `pair_manifest_sha256`,
  `registration_checkpoint_sha256`, `registration_report_sha256`,
  `export_spec_sha256`, `caption_spec_sha256`, `segmentation_spec_sha256`;
- a nonempty `records` list in explicit training order.

Each record contains a unique `pair_id`, official-train `sequence`, zero-based
`visible_frame_index` and `infrared_frame_index`, `height`, `width`, a frozen full
`prompt`, and four assets: `rgb`, `ir`, `seg`, `support`. Each asset is an object
with a relative `path` and byte `sha256`, inside the manifest's directory.
The two frame indices are separate: synchronized pairing is not assumed from
identical frame numbers. Duplicate frame pairs are rejected even with different IDs.

All four assets must have the declared, identical observation-grid dimensions:

| Asset | Required representation | Loader treatment |
|---|---|---|
| RGB | 8-bit RGB, already registered to the IR-domain grid | Bilinear resize |
| IR | 8-bit grayscale or RGB display image on that grid | Grayscale replicated to three channels; bilinear resize |
| Segmentation | 8-bit RGB rendering on that same grid | Nearest-neighbour resize, preserving palette |
| Support | 8-bit grayscale; every pixel equals 255 | Verified before loading the batch |

The support requirement is specific to an **unmasked** diffusion objective, not a
new registration-pass threshold. A registered common-view crop may be suitable,
but its source map, crop transform, observed support and target retention must be
exported and qualified explicitly. No automatic crop or ROI cherry-picking is
performed by this loader. If full-frame partially observed supervision is required,
the model loss must first implement and validate support masking; fabricating an
all-valid mask is not a solution. A crop-level result must not be reported as
full-frame registration qualification.

Output height and width are explicit multiples of 64. The entire input grid is
resized without a crop or flip; different aspect ratios imply documented x/y
scales. Image tensors use CHW float32 in [-1, 1]. This is display-intensity
normalization, **not a claim of calibrated thermal radiance**. No per-image
contrast fitting or silent uint16 truncation occurs. FLIR uint16 data needs its
separate frozen preprocessing policy and is not accepted as an implicit input.

The exporter must bind the actual alignment, captions and segmentation products
to these identities. Hash-shaped strings are not proof those external artifacts
exist or passed qualification. This low-level loader checks the conditioning
manifest, official train list, image bytes and shapes, **not the external
registration report's scientific validity or exporter correctness**. The caller
must validate that evidence before training. No current failed report is accepted
as authorization just because a batch can be loaded.

## Use and checks

After a qualified export exists, the dataset can be instantiated directly or used
as the upstream config's `data.params.train.target`:

```python
from aero_ir.generate.diffv2ir_data import DiffV2IRTrainDataset

dataset = DiffV2IRTrainDataset(
    manifest_path=export_manifest_path,
    train_split_path=official_train_json,
    expected_manifest_sha256=frozen_export_sha256,
    expected_train_split_sha256=frozen_train_sha256,
    output_size=(256, 256),
)
batch_item = dataset[0]
provenance = dataset.sample_provenance(0)
```

The variables above must come from a frozen experiment specification; do not
recompute the expected hashes from possibly changed files at every training
restart. The upstream trainer's validation configuration and checkpoint policy
also need explicit adaptation; replacing this one dataset target is **not** a
complete training command. This component reads no validation/test sequences.

CPU tests:

```bash
.venv/bin/python -m pytest -q tests/test_diffv2ir_data.py tests/test_diffv2ir_sampling.py tests/test_generator_evidence_gate.py
```

Tests cover batch collation, shape/range, synchronized geometry, segmentation
palette, deterministic loading, split leakage rejection, duplicate pairs,
provenance fields, pinned metadata, file drift, paths/symlinks, observation
support, and uint16 rejection. They do not establish model compatibility in the
full upstream runtime, successful diffusion training, registration accuracy, or
detector benefit. Sampling tests use an explicitly labelled identity test double
to exercise request/worker/image transport, padding restoration, failure handling,
output tampering and refusal of unsupported qualification/label claims. Those
test-double outputs are not synthetic-IR results.

The evidence gate was also checked against the actual saved v4 train/validation
screen and141,816-pair train-cache manifest. The zero-image, plan-only diagnostic
under `experiments/generator_evidence_gate_cpu_check_01/` records
`blocked_by_unqualified_evidence`; no external command or model was launched.
Plan directories are now fresh-only so an earlier evidence plan cannot be silently
overwritten. These checks establish internal consistency of saved evidence only:
they do not replay checkpoint inference, hash every cache array or qualify pixels.

## Actual upstream CPU runtime check

The dedicated Python 3.11 environment uses Torch 2.2.1+cu121, NumPy 1.26.4,
Lightning 1.9.5 and Transformers 4.26.1. Direct pins are in
`requirements/diffv2ir-runtime.txt`; the probe records all installed distribution
versions. This is not yet a portable, hash-locked installation recipe.

The pinned `CompVis/taming-transformers` revision
`3ba01b241669f5ade541ce990f7650a3b8f65318` has namespace-package directories
without `__init__.py`, but its setup script uses `find_packages()`. In the tested
build this produced an empty ordinary wheel containing metadata but no `taming`
modules. The unmodified checkout is instead installed in setuptools compatibility
editable mode. Do not repair this by changing upstream source or by modifying the
registration environment.

For a **new** dedicated environment and source directory, the setup sequence is:

```bash
python3.11 -m venv experiments/external/diffv2ir-runtime
experiments/external/diffv2ir-runtime/bin/python -m pip install -r requirements/diffv2ir-runtime.txt
git clone https://github.com/CompVis/taming-transformers.git experiments/external/taming-transformers
git -C experiments/external/taming-transformers checkout --detach 3ba01b241669f5ade541ce990f7650a3b8f65318
experiments/external/diffv2ir-runtime/bin/python -m pip install --no-deps --no-build-isolation --config-settings editable_mode=compat -e experiments/external/taming-transformers
```

These directories already exist in this workspace: do not rerun the creation
sequence over the installed runtime. The separate official DiffV2IR checkout
must also exist at the pinned revision above. No public model weights or CLIP
assets are downloaded by the CPU probe.

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  experiments/external/diffv2ir-runtime/bin/python -m scripts.check_diffv2ir_runtime \
  --out experiments/diffv2ir_runtime_cpu_probe_03.json
```

Use a fresh report filename for each rerun. The first recorded probe failed
`pip check` because project dependencies exposed by `PYTHONPATH=src` were missing.
After installing the explicit pandas/matplotlib/Hydra pins, report
`experiments/diffv2ir_runtime_cpu_probe_02.json` passed:

- `pip check`: no broken requirements;
- actual pinned upstream diffusion, autoencoder, encoder and UNet imports;
- one optimizer step through a small randomly initialized upstream UNet, with
  finite nonzero gradients and changed finite output;
- four analytic checks of the official CFG branch weighting;
- unchanged pinned DiffV2IR and taming source identities.

After adding the sampler's editable-dependency provenance check, the same actual
CPU probe was rerun successfully as
`experiments/diffv2ir_runtime_cpu_probe_03.json`. Its recorded probe-source hashes
match the current files. The full project CPU suite passed 540 tests, and Ruff
passed. The issued registration pilot source hashes still match their earlier
CPU-smoke snapshots; this runtime work does not alter either training arm.

The UNet is a small component fixture, and the CFG inputs are analytic fixtures.
Neither is a trained generator, a real thermal image, a full diffusion training
step, or evidence of registration quality. Full pretrained-model loading and GPU
inference remain untested; `generator_training_eligible` remains
`hold_not_qualified`.

## Full training-path CPU probe and Lightning hook repair

The earlier UNet-only component check did not exercise `Trainer.fit()` or the
VAE/text-conditioning/diffusion-loss connection. The first full-path probe,
`experiments/diffv2ir_training_cpu_probe_01/report.json`, failed before its first
optimizer update with:

```text
TypeError: LatentDiffusion.on_train_batch_start() missing 1 required positional argument: 'dataloader_idx'
```

The pinned upstream `LatentDiffusion` requires three hook arguments, while the
installed Lightning1.9.5 single-loader training loop supplies two.
`src/aero_ir/generate/diffv2ir_training.py` adds a mixin supplying the omitted
index as0 and forwarding to the unchanged upstream implementation. Its config
factory `create_training_model` requires Lightning1.9.5; callers must first
verify/import the pinned upstream. It adds no parameters, loss modifications,
qualification bypass or edits to the external checkout. It is a compatibility
component, not a registered-export authorizer or real-data training launcher.

With that adapter, the actual CPU rerun passed under
`experiments/diffv2ir_training_cpu_probe_02/report.json`:

- The project dataset loads four explicitly synthetic64x64 image triplets and
  full-support masks, with local hashed fixture provenance. No Anti-UAV/FLIR
  frames or pretend registration-pass records are used.
- Genuine upstream `AutoencoderKL`, `FrozenCLIPEmbedder`, `LatentDiffusion` and
  UNet components are instantiated at small test sizes. The local CLIP encoder
  is randomly initialized with a tiny tokenizer; it is not downloaded/pretrained
  semantic evidence. The denoiser has about0.80 million parameters.
- The batch becomes a `[1,4,32,32]` target latent, two matching image/segmentation
  conditioning latents, and `[1,32,32]` text features. The upstream diffusion loss
  produces finite nonzero gradients; frozen VAE/text parameters receive none.
- Lightning executes four microbatches with accumulation2: two optimizer updates.
  Saved checkpoint tensors reload exactly. Restarting via Lightning's checkpoint
  path executes two more microbatches and advances both global step and all Adam
  parameter step counters to3. Frozen encoder weights remain unchanged.
- This is optimizer-state continuation, **not** bitwise uninterrupted/resumed
  reproducibility. Random-state/data-loader equivalence has not been established.
- Upstream EMA updates on every `on_train_batch_end`: this probe records6 EMA
  updates for3 optimizer updates. That behaviour is preserved and explicit;
  future accumulation schedules must not describe it as optimizer-step EMA.

The probe runs offline and only on CPU, in the separate environment:

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  experiments/external/diffv2ir-runtime/bin/python -u \
  -m scripts.check_diffv2ir_training_runtime \
  --out-dir experiments/diffv2ir_training_cpu_probe_03
```

Use a fresh output directory. The `_01` failure and `_02` success are preserved;
`_03` above is a rerun example, not an already completed experiment. No GPU model
training, public-checkpoint compatibility, IR generation quality, physical
registration accuracy or generator-training authorization follows from this
small integration check. Its report explicitly records all those limitations.

## Remaining work before a GPU generator command

1. Review V7 results and complete registration qualification on the actual
   supervision domain without changing the existing pass thresholds.
2. Implement the registered colour-frame/conditioning exporter with real
   correspondence, support, crop/label and caption/mask provenance.
3. Test the pinned upstream model in a separate, locked runtime with a permitted
   initialization checkpoint; verify real inference and integrate the training
   launcher, including restart behaviour.
4. Run a real GPU forward/backward and fixed inference fixture before proposing
   the matched real-only / real+generated / real+simulated detector experiments.

This work does not replace registration repair, shorten the final experiment, or
claim a three-arm result. No generator GPU launch is ready yet.

## Isolated runtime compatibility work

`requirements/diffv2ir-runtime.txt` pins candidate direct dependencies for Python
3.11, including Torch 2.2.1, Lightning 1.9.5 and Transformers 4.26.1, plus a fixed
upstream taming-transformers revision. These differ from the author's requirements
and are **not yet a fully tested transitive lock**. Never apply this file to the
registration environment. The optional environment and pip download cache live in
git-ignored `experiments/external/`. The installation report records downloaded
package identities; a passing runtime probe is required separately.

After that separate installation succeeds, the offline, CPU-only compatibility
probe is:

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  experiments/external/diffv2ir-runtime/bin/python \
  -m scripts.check_diffv2ir_runtime \
  --out experiments/diffv2ir_runtime_cpu_probe_01.json
```

This imports the actual pinned diffusion, autoencoder, conditioning and UNet
modules, checks package consistency, runs a small **randomly initialized** upstream
UNet forward/backward/optimizer step, and tests the upstream CFG branch ordering
against known arithmetic fixtures. It downloads no weights, touches no experiment
data and does not run CUDA. A pass establishes only those component operations,
not compatibility with the full pretrained model, quality of generated IR,
diffusion training, registration qualification or an approved GPU experiment.
Reports retain errors on failure and refuse to overwrite prior evidence.

## Official checkpoint catalog (not an experiment selection)

The official README links to
[the author's Hugging Face distribution](https://huggingface.co/datasets/Lidong26/IR-500K/tree/fa0b38d18c8326e87f7de135869b8717068fd7dd/IR-500k/finetuned_checkpoints).
Its fixed-revision LFS metadata is recorded in
`configs/generator/diffv2ir_public_assets.json`: stage 1 is 2,132,869,203 bytes;
stage 2 and M3FD are 7,704,016,743 bytes each; FLIR is 7,704,016,434 bytes.
The catalog hashes are publisher metadata, not claims that model bytes were
downloaded and locally verified. No checkpoint is selected automatically.

The distribution currently provides no model card establishing permitted weight
uses or the exact data exposure of each checkpoint. These remain unresolved;
the upstream source-code license alone is not a review of the model assets.
In particular, a FLIR-named model cannot be presumed free of FLIR evaluation
exposure, and a stage-2 model cannot be presumed free of Anti-UAV exposure without
checking its training data. An engineering load test and a benchmark-approved
initialization are different decisions.
