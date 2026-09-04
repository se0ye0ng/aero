"""YOLOX training and inference behind a stable interface."""

from __future__ import annotations


class YOLOXDetector:
    def __init__(self, arch: str = "yolox-s", input_size=(640, 640), num_classes: int = 80,
                 pretrained: bool = False, weights: str | None = None) -> None:
        self.arch = arch
        self.input_size = tuple(input_size)
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.weights = weights

    def fit(self, train_set, val_set, train_cfg):
        """TODO: wrap the upstream trainer. Augmentation is fixed by protocol across arms."""
        raise NotImplementedError

    def predict(self, images):
        raise NotImplementedError
