# Registration v5 pilot: control and contingency

The running v5 pilot is a candidate repair, not a qualification result. Its ten
epochs must be evaluated with the same frame IDs, coordinate convention and
per-frame v4 engineering statistics as the initial v4 checkpoint. A falling
cycle loss alone cannot authorize a longer run: the identity map has perfect
cycles and topology without aligning two cameras.

## Parallel control

Run `bash scripts/run_antiuav300_registration_v5_control.sh` on a separate GPU
allocation. The wrapper preserves the scheduler's `CUDA_VISIBLE_DEVICES` setting.
On a node where GPU 1 is explicitly allocated and free, select it with
`CUDA_VISIBLE_DEVICES=1 bash scripts/run_antiuav300_registration_v5_control.sh`.

This experiment starts from the same completed v4 checkpoint and performs the
same ten additional epochs, batch size 8, seed 0, rotating train-pair schedule,
AdamW learning rate 1e-5 and cosine schedule to 1e-6 as the v5 pilot. It retains
the v4 loss. With 160 sequences and 16 pairs per sequence per epoch, each arm
uses 3,200 optimizer updates. The control retains the loader's eval mode to
match the existing v5 trainer; its gradient computation remains enabled.

Compare three frozen checkpoints: initial v4, v4-loss continuation, v5 repair.
Evaluate paired differences in box IoU, global/ROI cycle p95, ROI max cycle,
positive-Jacobian fraction, observed support, and joint frame pass rate. Include
per-sequence summaries and uncertainty aggregated by sequence, not independent
frame bootstraps. Training totals have different definitions across objectives
and must not be compared numerically. A single matched seed establishes only a
pilot comparison; a robust improvement claim needs further seeds.

The control writes only to
`experiments/antiuav300_registration_v5_control_e10_seed0` by default and refuses
to overwrite it. Both jobs can read the complete shared cache, but their CPU
memory and Lustre bandwidth requirements remain additive. No GPU job was
launched while preparing these scripts.

## Contingency decisions

1. **Cycle improves while box alignment deteriorates.** Quantify the tradeoff
   against both initial v4 and its matched continuation. Test a constrained
   repair objective that preserves train box alignment while reducing cycle
   tails. Fix any preservation margin using train-only calibration before
   evaluating held-out sequences. This is a new experiment, not an instruction
   to extend v5 automatically.
2. **Reciprocal cycle remains poor despite preserved alignment.** Test a shared
   invertible parameterization: a global affine map with an analytic inverse,
   then, only if necessary, an integrated stationary-velocity residual whose
   inverse uses the negative velocity. Numeric interpolation still requires the
   same cycle/Jacobian checks. A low cycle by construction is not evidence of
   physical alignment. This architecture is proposed, not implemented here.
3. **Support dominates failures.** Measure actual camera overlap independently.
   Two distinct fields of view need not have 90% bidirectional overlap. First
   test feasibility using train-only camera/landmark evidence; do not force the
   learned map toward identity to satisfy unsupported overlap assumptions. Any
   revised overlap mask or scope needs its own frozen protocol and new held-out
   evaluation, retaining the original results and thresholds in the record.
4. **Box alignment plateaus while geometry improves.** Audit frame timing,
   camera motion, differing silhouettes, small-target annotation uncertainty
   and occlusion on a fixed, blinded correspondence panel. A box corner is not
   a corresponding physical landmark. Model capacity cannot repair incorrect
   or unobservable supervisory correspondences.
5. **Engineering geometry passes.** Collect independent physical correspondence
   and occlusion evidence before allowing paired generator training. Current
   `qualification_v4.gate_report` deliberately retains this separate HOLD;
   neither additional epochs nor an engineering pass clears it.

The pilot must be screened by an evaluator that accepts completed pilot
metadata. The existing `audit_antiuav300_registration_v4.py` requires an actual
300-epoch checkpoint and must not be used by relabeling a ten-epoch checkpoint
as epoch 300. Keep all evaluations diagnostic until the independent evidence
and intended full-image versus target-only scope have been established.
