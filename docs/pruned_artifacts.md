# Pruned regenerable artifacts (2026-10-02)

The following regenerable caches were removed before the public release to reduce
local storage. None of them is a training result: every entry is either a derived
cache with a retained content-addressed manifest or an incomplete download. The
trained checkpoints, audit reports and figures were not touched.

## Removed

| Path | Size | Recovery |
|---|---|---|
| `experiments/antiuav300_registration_v2_full_train_cache/shards/` | 52 GB | `scripts/run_antiuav300_registration_v2.sh` rebuilds it from `AERO_ANTIUAV300_ROOT`; `manifest.json` is retained and carries the per-shard SHA-256 so the rebuild is verifiable |
| `experiments/external/ms2_first_train_archives/*.tar.bz2.part` | 42 GB | Incomplete resumable downloads, never fully retrieved. Re-fetch with the upstream MS² download scripts (see `scripts/download_ms2.sh`) |
| `experiments/external/pip-cache/` | 2.9 GB | Python wheel cache; repopulated by `pip install -r requirements/*.txt` |

## Retained

- All trained checkpoints under `experiments/yolox_runs/` and the registration training runs.
- All audit reports, manifests and `*.json` run records.
- All diagnostic figures.
- `antiuav300_registration_v2_full_train_cache/manifest.json` (160 sequences, 141,816 pairs,
  cache manifest SHA-256 `3fc1bff8...b3b560e`).
