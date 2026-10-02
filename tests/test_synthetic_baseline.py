"""Mechanical checks for the procedural baseline, not physical realism certification.

Soft edges remain box-conditioned. These checks do not exclude label shortcuts or
establish equivalence to thermal simulation.
"""

import numpy as np
import pytest
from scipy import ndimage

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


@pytest.mark.parametrize("labels", [[], ["car", "person"]])
def test_class_count_mismatch_cannot_silently_drop_objects(labels):
    with pytest.raises(ValueError, match="exactly one class"):
        SyntheticBaselineGenerator().render(_rgb(), [(10, 10, 8, 8)], labels)


@pytest.mark.parametrize(
    "boxes",
    [
        [[1, 2, 3]],
        [[1, 2, np.nan, 4]],
        [[1, 2, -3, 4]],
        [[-1, 2, 3, 4]],
        [[90, 2, 8, 4]],
        [[1, 2, 0.1, 4]],
    ],
)
def test_invalid_or_unrenderable_boxes_are_rejected(boxes):
    with pytest.raises(ValueError):
        SyntheticBaselineGenerator().render(_rgb(), boxes)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2, 2)),
        np.zeros((0, 2, 3)),
        np.full((2, 2, 3), np.nan),
        np.array(1),
    ],
)
def test_invalid_images_are_rejected(image):
    with pytest.raises(ValueError, match="image"):
        SyntheticBaselineGenerator().render(image, [])


def test_valid_rendering_is_unchanged_and_empty_scene_supported():
    gen = SyntheticBaselineGenerator()
    rgb = _rgb()
    lum = gen._luminance(rgb)
    background = gen.base_temp_k + gen.scene_contrast_k * ndimage.gaussian_filter(
        lum, gen.background_smooth_px
    )
    target = np.zeros(lum.shape)
    target[10:18, 10:18] = gen.class_delta_k["car"]
    expected = (
        background + ndimage.gaussian_filter(target, gen.edge_softness_px)
    ) * gen.scene_response_dn_per_K
    np.testing.assert_array_equal(gen.render(rgb, [(10, 10, 8, 8)], ["car"]), expected)
    np.testing.assert_array_equal(gen.render(rgb, [], []), background * gen.scene_response_dn_per_K)


def test_generate_rejects_mismatched_annotations_instead_of_returning_false_labels():
    with pytest.raises(ValueError, match="exactly one class"):
        SyntheticBaselineGenerator().generate([_rgb()], [{"boxes": [(10, 10, 8, 8)], "labels": []}])
