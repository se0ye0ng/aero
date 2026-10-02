"""Fetch predetermined panel; Seaside/6 has no authored landmark files."""

import sys
from unittest.mock import patch

import yaml

from scripts import fetch_registration_landmarks as fetcher


def main():
    path = "configs/experiment/registration_roma_fresh_confirmation.yaml"
    with open(path) as handle:
        config = yaml.safe_load(handle)
    select = fetcher.select_samples

    def select_count(entries, count):
        parents = {e["path"].split("/")[0] for e in entries}
        if len(parents) != 1:
            raise ValueError("ambiguous scene inventory")
        scene = next(iter(parents))
        return select(entries, config["sample_count_overrides"].get(scene, count))

    with (
        patch.object(fetcher, "select_samples", select_count),
        patch.object(sys, "argv", [fetcher.__file__, "--config", path]),
    ):
        fetcher.main()


if __name__ == "__main__":
    main()
