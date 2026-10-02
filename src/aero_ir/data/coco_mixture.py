"""Deterministic, without-replacement COCO mixing with explicit origin identities.

This assembles already authorized pools; it does not authorize generated data or
establish split independence. Callers must verify source artifacts and eligibility.
"""

from copy import deepcopy

import numpy as np


def indexed(pool):
    images, categories, annotations = {}, {}, {}
    for name, target in (
        ("images", images),
        ("categories", categories),
        ("annotations", annotations),
    ):
        for item in pool[name]:
            key = item["id"]
            if type(key) is not int or key in target:
                raise ValueError(f"invalid or duplicate {name} ID")
            target[key] = item
    grouped = {key: [] for key in images}
    for ann in annotations.values():
        if ann["image_id"] not in images or ann["category_id"] not in categories:
            raise ValueError("annotation references an unknown image or category")
        image = images[ann["image_id"]]
        box = np.asarray(ann["bbox"], dtype=float)
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or (box[:2] < 0).any()
            or (box[2:] <= 0).any()
            or (box[:2] + box[2:] > [image["width"], image["height"]]).any()
        ):
            raise ValueError("invalid or out-of-image annotation")
        grouped[ann["image_id"]].append(ann)
    return images, categories, grouped


def assemble_mixture(real, auxiliary, *, n_real, n_auxiliary, seed):
    if any(type(v) is not int or v < 0 for v in (n_real, n_auxiliary, seed)):
        raise ValueError("nonnegative integer counts and seed required")
    if n_real + n_auxiliary == 0:
        raise ValueError("empty mixture is not a training set")
    pools = [indexed(real), indexed(auxiliary)]
    names = [{k: v["name"] for k, v in pool[1].items()} for pool in pools]
    if names[0] != names[1]:
        raise ValueError("class ID/name mappings differ across pools")
    result = dict(
        images=[],
        annotations=[],
        categories=deepcopy(real["categories"]),
        info=dict(selection_seed=seed, replacement=False),
    )
    origins = []
    for index, (count, pool, role) in enumerate(
        zip((n_real, n_auxiliary), pools, ("real", "auxiliary"), strict=True)
    ):
        images, _, annotations = pool
        if count > len(images):
            raise ValueError(f"insufficient {role} images")
        # Same real permutation across all ratios gives nested real selections.
        keys = sorted(images)
        order = np.random.default_rng(np.random.SeedSequence([seed, index])).permutation(len(keys))
        for offset in order[:count]:
            original_id = keys[offset]
            new_id = len(result["images"]) + 1
            image = deepcopy(images[original_id])
            image["id"] = new_id
            result["images"].append(image)
            origins.append(dict(image_id=new_id, pool=role, source_image_id=original_id))
            for source_ann in sorted(annotations[original_id], key=lambda a: a["id"]):
                annotation = deepcopy(source_ann)
                annotation.update(id=len(result["annotations"]) + 1, image_id=new_id)
                result["annotations"].append(annotation)
    return result, origins
