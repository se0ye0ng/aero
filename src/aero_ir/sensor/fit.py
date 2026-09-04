"""Fit sensor parameters to a real image set.

Used by ``configs/sensor/matched.yaml``. The identifiable parameters are those with a
distinct signature in the radiometric statistics: NETD from the noise PSD floor, MTF cutoff
from the radial power spectrum roll-off, column FPN from the column/row variance ratio, and
the AGC clip points from the intensity histogram.

Not every parameter is identifiable from imagery alone; the fit reports a confidence per
parameter and falls back to the configured default where identification fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SensorFitResult:
    params: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)


def fit_sensor_params(real_images, defaults: dict[str, float]) -> SensorFitResult:
    """Estimate sensor parameters from a real image set.

    TODO(N2): implement. Planned estimators:
      - netd_mK                  from the high-frequency plateau of the noise PSD (R5)
      - cutoff_cycles_per_pixel  from the -3 dB point of the radial power spectrum (R4)
      - column_sigma_dn          from the column/row variance ratio of the residual (R8)
      - low/high_percentile      from the clipped-pixel fraction of the histogram (R7)
    """
    raise NotImplementedError("N2 track - see docs/roadmap.md")
