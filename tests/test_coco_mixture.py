from copy import deepcopy

import pytest

from aero_ir.data.coco_mixture import assemble_mixture


def pool():
    return dict(
        images=[dict(id=i, file_name=f"{i}.tiff", width=10, height=10) for i in range(3)],
        categories=[dict(id=8, name="truck")],
        annotations=[dict(id=1, image_id=0, category_id=8, bbox=[1, 2, 3, 4], area=12)],
    )


def test_collision_free_deterministic_and_preserves_negative_images():
    source = pool()
    before = deepcopy(source)
    first, origins = assemble_mixture(source, source, n_real=3, n_auxiliary=3, seed=0)
    assert first == assemble_mixture(source, source, n_real=3, n_auxiliary=3, seed=0)[0]
    assert len(first["images"]) == 6 and len(first["annotations"]) == 2
    assert len({i["id"] for i in first["images"]}) == 6
    assert all(a["bbox"] == [1, 2, 3, 4] and a["category_id"] == 8 for a in first["annotations"])
    assert len(origins) == 6 and source == before


def test_real_subsets_are_nested_for_fixed_total_comparison():
    _, full = assemble_mixture(pool(), pool(), n_real=3, n_auxiliary=0, seed=2)
    _, mixed = assemble_mixture(pool(), pool(), n_real=2, n_auxiliary=1, seed=2)
    assert [x["source_image_id"] for x in full[:2]] == [x["source_image_id"] for x in mixed[:2]]


def test_class_mismatch_and_dangling_annotation_rejected():
    other = pool()
    other["categories"][0]["name"] = "car"
    with pytest.raises(ValueError, match="mappings"):
        assemble_mixture(pool(), other, n_real=1, n_auxiliary=1, seed=0)
    other = pool()
    other["annotations"][0]["image_id"] = 99
    with pytest.raises(ValueError, match="unknown"):
        assemble_mixture(pool(), other, n_real=1, n_auxiliary=1, seed=0)


@pytest.mark.parametrize("count", [-1, 1.5, True, 4])
def test_invalid_or_insufficient_count_rejected(count):
    with pytest.raises(ValueError):
        assemble_mixture(pool(), pool(), n_real=count, n_auxiliary=1, seed=0)
