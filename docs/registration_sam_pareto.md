# Non-deteriorating silhouette alignment: completed CPU ablation

## Outcome

The weighted Dice/Chamfer objective can sacrifice boundary alignment for overlap.
Explicit non-deterioration prevents that failure, but **does not solve physical
registration**. On the already observed16-sequence train panel, the problem actual
pair returns to identity; no improved transform was found within the tested pool.
Generator eligibility and every v4 qualification threshold remain unchanged.

Authoritative report: `experiments/registration_sam_pareto_train16_02/report.json`.
The earlier`_01` is retained as the initial execution before stricter input validation
and formatting;`_02` uses the finalized sources, archived with hashes in its directory.
Parent: the fresh GPU `experiments/registration_sam_train16_v2/report.json`, not the
cached-mask reprojection. No new SAM inference, images, annotations, or GPU work.

## Method

For each previously eligible safe fit, form one deterministic candidate pool:
identity, the saved v2 optimum, and its3^4 local neighbours (scale factor1.05,
angle±3degrees, translations±2 crop pixels). Retain the original hard bounds.
Both methods evaluate the **same pool** and the same original-prompt masks at256px:

1. Reference: minimize symmetric Dice loss + symmetric Chamfer/256.
2. Constrained: minimize the same objective only among candidates whose Dice and
   Chamfer are each no worse than the safe identity baseline (numerical tolerance1e-10).

Missing safe identity causes abstention, not a fabricated improvement. Identity
fallbacks are counted separately and are not successful corrections. Original
ineligible cases remain in the16-case denominator. No pair label, perturbed prompt,
validation/test image or box score ranks candidates. Existing train boxes were
already used to define SAM inputs, so this is not an annotation-free method.
Candidates, masks' source hashes, transforms and analytic inverses are recorded.
The fixed selected transform is then checked on two perturbed prompt masks without
refitting. These checks are correlated model predictions, not independent GT.

This is local refinement, not a full constrained global search or a proof that no
non-deteriorating transform exists elsewhere. The same budgets apply to real and
shuffled pairs. The panel was inspected in previous development, not held out.

## Results

| Group / method | Available /16 | Identity fallbacks | Comparable baseline count | Dice loss median before→after | Chamfer median before→after | Chamfer regressions |
|---|---:|---:|---:|---:|---:|---:|
| Actual / reference | 8 | 0 | 7 | 0.244876→0.161115 | 2.949863→1.336335 | 1 |
| Actual / constrained | 7 | 1 | 7 | 0.244876→0.161115 | 2.949863→1.336335 | 0 |
| Shuffled / reference | 7 | 0 | 7 | 0.566447→0.264449 | 5.622722→3.572008 | 2 |
| Shuffled / constrained | 7 | 2 | 7 | 0.566447→0.264449 | 5.622722→3.572008 | 0 |

Distances are corrected-crop pixels; lower is better. Medians do not expose the
tail failure: removing regressions does not change either median. Constrained
actual corrections are6/16, not7/16 or100%. One formerly fitted actual case has
no safe identity baseline and now abstains. The constrained common-source subset
is five RGB cases: actual Dice wins4/5 and Chamfer wins3/5. The reference subset
is six cases: Dice wins5/6 and Chamfer wins3/6. These changing cohorts do not
demonstrate stronger real/negative separation.

On perturbed prompts, Chamfer regressions fall from2/14 to0/14 actual observations,
and4/14 to0/14 shuffled observations. This is useful numerical protection, not a
physical accuracy certificate. The known synthetic16px translation test retains
zero Dice/Chamfer error and the correct shared inverse under the new rule.

## Reproduce (CPU only)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.probe_registration_sam_pareto \
  --source-report experiments/registration_sam_train16_v2/report.json \
  --output-dir experiments/registration_sam_pareto_repeat01
```

The output directory must not already exist. All parent artifacts are hash-checked
before use. This command does not change the previous source, checkpoint, output,
pairing, or qualification files. Invalid/replayed inputs fail closed.

## Consequence for the full pipeline

- Coordinate correctness: independently checked, but not an accuracy claim.
- Silhouette non-deterioration: implemented and measured, insufficient as a sole anchor.
- Learned dense geometry: still fails the frozen gate. V6 predicts its two fields
  independently; preservation penalties do not make them a shared invertible map.
- Physical correspondence: still unestablished; neither a perfect cycle nor a
  stable SAM mask proves same-surface pixels. Human annotation is not a prerequisite
  to developing a repair, but proxies must not be relabeled as measured accuracy.
- Paired generator and three-arm detector experiment: not complete; do not launch
  or publish an efficacy claim as though these diagnostics completed those stages.

The next trained model intervention must change native alignment and enforce a shared
inverse jointly, rather than only reconstructing the reciprocal of the fixed v6
field (already bounded below the95% geometry requirement). A train-only feasibility
test of a shared smooth transform with image-based anchors should precede another
long training run. The [shared-velocity primitive and initialization check](registration_shared_velocity.md)
are now implemented: naive displacement reuse improves cycles but worsens alignment,
so it is not a ready training/deployment solution. Any successful candidate still needs the unchanged geometric
screen and independent correspondence evidence on its actual supervision domain.
