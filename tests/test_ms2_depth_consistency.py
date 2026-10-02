import numpy as np
import pytest

from aero_ir.data.ms2_depth_consistency import depth_product_consistency


def test_known_axial_geometry_and_range_alternative():
    source = np.zeros((3, 3), dtype=np.uint16)
    target = source.copy()
    source[1, 1], target[1, 1] = 256, 512
    transform = np.eye(4)
    transform[2, 3] = 1
    result = depth_product_consistency(source, target, np.eye(3), np.eye(3), transform)
    axial = result["conventions"]["axial_z"]
    radial = result["conventions"]["euclidean_range"]
    assert axial["associated_absolute_depth_residual_m"]["max"] == 0
    assert radial["associated_absolute_depth_residual_m"]["min"] > 0.1
    assert axial["associated_fraction_of_source"] == 1.0
    assert result["independent_image_reference"] is False
    assert result["correspondence_identity_verified"] is False
    assert result["registration_qualified"] is False


@pytest.mark.parametrize("side", ["source", "target", "both"])
def test_missing_depth_stays_missing(side):
    source = np.full((3, 3), 256, dtype=np.uint16)
    target = source.copy()
    if side in ("source", "both"):
        source[:] = 0
    if side in ("target", "both"):
        target[:] = 0
    result = depth_product_consistency(source, target, np.eye(3), np.eye(3), np.eye(4))
    for item in result["conventions"].values():
        assert item["associated_points"] == 0
        assert item["associated_absolute_depth_residual_m"]["median"] is None


def test_depth_error_and_off_image_points_are_not_hidden():
    source = np.full((3, 3), 256, dtype=np.uint16)
    target = np.full((3, 3), 512, dtype=np.uint16)
    result = depth_product_consistency(source, target, np.eye(3), np.eye(3), np.eye(4))
    axial = result["conventions"]["axial_z"]
    assert axial["associated_absolute_depth_residual_m"]["median"] == 1.0
    transform = np.eye(4)
    transform[0, 3] = 100
    result = depth_product_consistency(source, target, np.eye(3), np.eye(3), transform)
    assert result["conventions"]["axial_z"]["projected_supported"] == 0
    assert result["conventions"]["axial_z"]["source_points"] == 9


def test_bad_radius_rejected():
    depth = np.ones((3, 3), dtype=np.uint16)
    with pytest.raises(ValueError):
        depth_product_consistency(
            depth, depth, np.eye(3), np.eye(3), np.eye(4), association_radius_px=0
        )
