"""Label-transfer audit (F4).

Labels are carried across the generation step. If generation displaces or deforms an object,
the box no longer marks the object, and the resulting degradation is attributed to
"generated data" when it belongs to the label pipeline. Nothing in the observed downstream
results measures this.

Every generated image is audited before it can enter a training set. Failures are excluded
and the exclusion rate is reported alongside every result.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LabelAuditRecord:
    image_id: str
    iou: float
    centroid_shift_frac: float
    area_ratio_change: float
    min_target_contrast: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class LabelAuditSummary:
    n_total: int = 0
    n_excluded: int = 0
    records: list[LabelAuditRecord] = field(default_factory=list)

    @property
    def exclusion_rate(self) -> float:
        return self.n_excluded / self.n_total if self.n_total else 0.0


def audit(generated_set, source_set, cfg) -> LabelAuditSummary:
    """Compare transferred boxes against boxes re-estimated on the generated image.

    TODO: implement. Planned checks, all configured in ``configs/data/*.yaml``:
      - IoU between the source box and a re-estimated box (segmentation or saliency based)
      - centroid displacement as a fraction of the box diagonal
      - target pixel-area ratio change
      - target-background contrast floor, in units of local background sigma
        (reuses :func:`aero_ir.rfs.stats.target_snr`, so the audit and the fidelity
        diagnostic share one definition of contrast)
    """
    raise NotImplementedError("see docs/experiment_protocol.md - label transfer audit")
