# Registration pilot comparison

Run from an allocated GPU node after the v4 baseline, v5 control, v5 seeds 0/1,
and v6 seed 0 checkpoints have completed:

```bash
bash scripts/run_antiuav300_registration_comparison.sh
```

The wrapper preserves the scheduler's `CUDA_VISIBLE_DEVICES`. It evaluates five
models sequentially within each batch on one GPU, decoding each RGB/IR pair once.
It never trains a model. All checkpoints and their common training-cache lineage
are checked before inference. The original v4 qualification evaluator is unchanged.

For a CPU-only input check (no inference or output directory creation):

```bash
bash scripts/run_antiuav300_registration_comparison.sh --preflight-only
```

The default output directory is
`experiments/antiuav300_registration_comparison_v4_v5_v6_01`. An existing directory
is rejected; use a fresh run ID for a repeat or a failed/interrupted evaluation:

```bash
AERO_REGISTRATION_COMPARE_OUTPUT="$PWD/experiments/antiuav300_registration_comparison_v4_v5_v6_02" \
  bash scripts/run_antiuav300_registration_comparison.sh
```

For less memory use, append `--batch-size 4`. Do not change batch size mid-run.
Progress appears after the first batch and every ten batches; video decoding and
annotation verification may take time before the first batch.

## Outputs and interpretation

- `summary.md`: readable comparison for each split and point-map direction.
- `summary.csv`: means, medians, missing measurement counts and pass rates.
- `report.json`: per-frame measurements, paired candidate-minus-v4 deltas,
  checkpoint/training metadata, source and annotation hashes, decoded-input hash,
  thresholds and gates. Its content hash excludes the `report_sha256` field.

Selection uses the same eight endpoint-inclusive usable pairs per sequence as the
v4 screen: 1,280 train and 536 validation pairs for the current release. Missing,
duplicate or unexpected frames fail the comparison. No test split is accessed.
Validation has been used during development; this is not independent confirmation.

`cycle_p95_pixels` is computed within each frame on valid round-trip support.
The table reports the **median of those per-frame p95 values**, in 256x256 network
pixels. It is neither a pooled percentile nor native RGB/IR pixel accuracy.
Invalid measurements are counted explicitly, excluded from numeric summaries,
and fail the unchanged engineering gate. Read support and missing counts alongside
cycle statistics. Nonpositive Jacobian fractions describe sampled cell-centre
topology and cannot establish physical correspondence correctness.

Higher box IoU is better; lower centroid/area errors, cycle errors, and nonpositive
Jacobian fractions are better. Evaluate these together on the same frames.
For each candidate, `paired_deltas_vs_baseline` reports candidate minus v4.
Pass rates use the existing v4 thresholds and require both directions on the same
frame. Successful exit (0) means files were produced, **not** qualification PASS.
The exhaustive and independent-correspondence requirements remain unsatisfied by
this sampled diagnostic, so generator eligibility remains HOLD even if geometric
screen criteria pass. Loss reductions alone do not authorize full training.

Custom subsets can be evaluated with repeated `--checkpoint NAME=PATH` arguments:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.compare_antiuav300_registration_pilots \
  --root /lustre/winston1214/dataset/Anti-UAV300 \
  --checkpoint "v4=$CHECKPOINT_V4" \
  --checkpoint "v6=$CHECKPOINT_V6" \
  --output-dir "$PWD/experiments/registration_v4_v6_comparison_01" \
  --device cuda
```

Set the two checkpoint variables to single-line, existing file paths first. Include
v4 in every subset so all deltas retain a common reference.
