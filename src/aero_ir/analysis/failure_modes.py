"""Stratified analysis (F5).

A single aggregate AP cannot say *which* failure mode moved, which is why studies that report
only AP terminate in a question mark. Every result in this repository is decomposed along
four axes before it is interpreted.
"""

from __future__ import annotations

STRATA = {
    "target_pixel_area": ["tiny(<16)", "small(<32^2)", "medium(<96^2)", "large"],
    "contrast_quartile": ["q1", "q2", "q3", "q4"],
    "polarity": ["hot_on_cold", "cold_on_hot"],
    "clutter_level": ["low", "medium", "high"],
}


def stratified_ap(predictions, ground_truth, strata=None) -> dict:
    """AP within each stratum, plus the change attributable to generated data per stratum."""
    raise NotImplementedError
