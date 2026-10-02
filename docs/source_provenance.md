# Source provenance of recorded runs

Every recorded run pins the SHA-256 of the sources that produced it, under `sources` in its
report. The guard is `aero_ir.utils.manifest.verify_recorded_sources`, exercised by
`tests/test_source_supersession.py`. **A recorded digest is never rewritten.** When a source no
longer matches, the run is not re-pointed at the new file; either the old revision is found, or
the run is declared unverifiable.

A pinned source resolves in one of three ways.

## 1. It still matches the working tree

The ordinary case, reported as `worktree`.

## 2. It matches a commit named in `docs/source_supersession.json`

The source was edited after the run. The run is verified against the commit recorded there,
which is the last commit whose content matches what the run recorded, and the verification
reports that commit instead of `worktree`.

One entry exists today, at commit `57561a3d71e6b96296a7a33fe7e7d50d9db351b8`: the public
release replaced hard-coded local dataset paths with the `aero_ir.utils.paths` helpers that
resolve `AERO_DATA_ROOT` and `AERO_*_ROOT`. Only argparse defaults and module constants moved.
Behaviour is identical whenever `--root` or the environment variables are set, which every
recorded run did, so no run's result is affected — but the files' hashes changed, and that is
what the record states.

## 3. It matches nothing

The source was edited between the run and the first commit containing it, so the revision that
produced the run was never committed and cannot be recovered. These runs are listed in
`UNRECOVERABLE_SOURCE_RUNS` in `tests/test_source_supersession.py` rather than hidden, and the
test asserts the list does not grow and that no entry has silently become verifiable again.

Ten runs are in that state, all predating the public release:

| Run | Unverifiable source |
|---|---|
| `detector_free_train2_01` | `scripts/probe_detector_free_matching.py` |
| `registration_mi_probe_train16_01` | `scripts/probe_registration_mi.py` |
| `registration_mi_probe_train16_02` | `scripts/probe_registration_mi.py` |
| `registration_mind_probe_train16_01` | `scripts/probe_registration_mind.py` |
| `registration_mind_probe_train160_fine_01` | `scripts/probe_registration_mind.py` |
| `registration_ngcc_train16_01` | `scripts/probe_registration_ngcc.py` |
| `registration_part_practice_v1` | `docs/registration_practice_ko.md` |
| `registration_sam_pareto_train16_01` | `scripts/probe_registration_sam_pareto.py` |
| `registration_sam_v2_cached_repair_01` | `scripts/probe_registration_sam_v2.py` |
| `xoftr_overlay_train16_01` | `scripts/probe_xoftr_overlay.py` |

**What this costs.** These runs' numbers cannot be tied to an inspectable source revision. They
remain usable as diagnostics and as a record of what was tried; they are not admissible where a
reproducible source chain is required, and none of them is load-bearing for a qualification
decision — the registration gate is held open on other grounds. `xoftr_overlay_train16_01` is
cited as a `source_report` by several experiment configs, so results derived from it inherit the
same limitation. Re-running these probes on the committed sources would clear the list.

**Avoiding new entries.** Commit a probe before recording a run with it. A run whose source is
only in the working tree has no recoverable provenance the moment that file is edited.
