"""Generator interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    """Any generator usable by the pipeline.

    Implementations must carry labels through explicitly rather than assuming the source
    boxes remain valid; whatever they return is what the label audit (F4) inspects.
    """

    name: str

    def generate(self, sources, labels, **kwargs):
        """Return ``(images, boxes_per_image, provenance)``.

        ``provenance`` records the source id, conditioning inputs and sampling parameters
        for every output, so any generated image can be traced back to what produced it.
        """
        ...
