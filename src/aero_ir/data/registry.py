"""Dataset registry.

Datasets are declared in ``configs/data/*.yaml`` and resolved here. Nothing is downloaded
automatically from a gated source; ``scripts/download_*.sh`` prints the access route and
verifies checksums after manual placement.
"""

from __future__ import annotations

DATASETS = {
    "flir_urban": "Teledyne FLIR ADAS Thermal v2 - reproduction anchor",
    "llvip": "LLVIP - aligned low-light visible-infrared pairs",
    "antiuav_small": "Anti-UAV410 - small low-contrast thermal targets",
    "dronevehicle": "DroneVehicle - aerial RGB-IR, oriented boxes",
}


def build_dataset(cfg):
    """Return ``(images, boxes_per_image, metadata)`` for a split.

    TODO: implement per-dataset loaders. Contract:
      - images are float arrays in a consistent intensity unit, shape (H, W)
      - boxes are (x, y, w, h) in pixels, COCO convention
      - metadata carries the keys named by ``data.held_out_scenario.key`` so the
        coverage split (F2) can be applied without dataset-specific code elsewhere
    """
    raise NotImplementedError
