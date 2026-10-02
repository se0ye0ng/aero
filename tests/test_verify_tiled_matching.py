import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.verify_antiuav_tiled_matching import (
    canonical_hashes,
    check_score,
    safe_artifact,
    validate_tile_inventory,
)


def test_absolute_and_relative_provenance_are_equivalent():
    name = "scripts/probe_antiuav_tiled_matching.py"
    assert canonical_hashes({name: "digest"}) == canonical_hashes(
        {str(Path(name).resolve()): "digest"}
    )


def test_conflicting_provenance_aliases_rejected():
    name = "scripts/probe_antiuav_tiled_matching.py"
    with pytest.raises(ValueError, match="conflicting provenance"):
        canonical_hashes({name: "first", str(Path(name).resolve()): "different"})


def flags():
    return dict(
        tile_pairs=81,
        before_exact_dedup=81,
        records=[dict(tile0=i, tile1=j, retained=1) for i in range(9) for j in range(9)],
    )


def test_complete_inventory():
    validate_tile_inventory(flags())


@pytest.mark.parametrize("change", ["missing", "duplicate", "count", "negative"])
def test_tile_inventory_tampering_rejected(change):
    f = copy.deepcopy(flags())
    if change == "missing":
        f["records"].pop()
    elif change == "duplicate":
        f["records"][-1] = f["records"][0]
    elif change == "count":
        f["before_exact_dedup"] += 1
    else:
        f["records"][0]["retained"] = -1
    with pytest.raises(ValueError):
        validate_tile_inventory(f)


@pytest.mark.parametrize("a,b", [(None, 0.0), (0.5, 0.6), (np.nan, 0.5), (np.inf, np.inf)])
def test_invalid_or_changed_scores_rejected(a, b):
    with pytest.raises(ValueError):
        check_score(a, b)


def test_tiny_roundoff_and_unsupported_preserved():
    check_score(0.5, 0.5 + 1e-10)
    check_score(None, None)


def test_parent_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        safe_artifact(tmp_path, dict(artifact="../outside.npz", sha256="unused"))
