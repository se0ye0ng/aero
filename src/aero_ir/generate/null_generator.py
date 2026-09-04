"""Identity generator.

Used to validate the pipeline end to end without a generative model, and as the control arm
for real-plus-real runs - the arm that shows how much of any measured effect comes from
changing the data volume rather than from the data being generated.
"""

from __future__ import annotations


class NullGenerator:
    name = "null_passthrough"

    def generate(self, sources, labels, **kwargs):
        provenance = [{"source_id": i, "generator": self.name} for i in range(len(sources))]
        return sources, labels, provenance
