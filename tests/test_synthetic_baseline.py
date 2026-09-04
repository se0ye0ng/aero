"""The simulated arm must produce radiometrically plausible targets, not rectangles.

If the baseline hands the detector a hard-edged box of constant intensity, it is not a
simulator - it is a label leak, and any comparison against it is meaningless.
"""

import numpy as np

from aero_ir.generate.synthetic_baseline import SyntheticBaselineGenerator
from aero_ir.rfs.stats import target_background_delta


def _rgb(h=96, w=96, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3)) * 0.4 + 0.3


def test_targets_are_warmer_than_background():
    gen = SyntheticBaselineGenerator()
    boxes = [(40, 40, 12, 12)]
    img = gen.render(_rgb(), boxes, ["car"])
    assert target_background_delta(img, boxes)[0] > 0


def test_class_offsets_are_respected():
    gen = SyntheticBaselineGenerator()
    boxes = [(40, 40, 12, 12)]
    rgb = _rgb()
    person = target_background_delta(gen.render(rgb, boxes, ["person"]), boxes)[0]
    bike = target_background_delta(gen.render(rgb, boxes, ["bike"]), boxes)[0]
    # person is configured 3 K warmer than bike; ordering must survive rendering
    assert person > bike


def test_target_edges_are_soft():
    """A hard rectangular boundary would be a trivially learnable artifact."""
    gen = SyntheticBaselineGenerator(edge_softness_px=1.5)
    img = gen.render(_rgb(), [(40, 40, 12, 12)], ["car"])
    row = img[46, 36:56]
    steps = np.abs(np.diff(row))
    # a hard edge concentrates the whole transition in one pixel; a soft one spreads it
    assert steps.max() < 0.8 * (row.max() - row.min())


def test_deterministic():
    gen = SyntheticBaselineGenerator()
    rgb, boxes = _rgb(), [(10, 10, 8, 8)]
    assert np.allclose(gen.render(rgb, boxes, ["car"]), gen.render(rgb, boxes, ["car"]))


def test_unknown_class_falls_back_to_default():
    gen = SyntheticBaselineGenerator()
    boxes = [(40, 40, 12, 12)]
    img = gen.render(_rgb(), boxes, ["something_unseen"])
    assert target_background_delta(img, boxes)[0] > 0
