from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.probe_registration_temporal_violation import read_config, temporal_schedule


def test_temporal_frames_are_distinct_balanced_and_exclude_probe_positions():
    shards = [{"sequence_id": f"s{i:03}", "pairs": 12 + i} for i in range(160)]
    batches = temporal_schedule(shards, 10, 8, 0)
    assert len(batches) == 200
    assert batches == temporal_schedule(list(reversed(shards)), 10, 8, 0)
    assert batches != temporal_schedule(shards, 10, 8, 1)
    lengths = {s["sequence_id"]: s["pairs"] for s in shards}
    pairs = [(sid, index) for batch in batches for sid, index in batch]
    assert len(set(pairs)) == 1600
    assert set(Counter(sid for sid, _ in pairs).values()) == {10}
    for sid, index in pairs:
        assert 0 <= index < lengths[sid]
        assert index not in {lengths[sid] // 2, lengths[sid] // 4}
    for batch in batches:
        assert len({sid for sid, _ in batch}) == 8
    for start in range(0, 200, 20):
        assert {sid for b in batches[start : start + 20] for sid, _ in b} == set(lengths)


@pytest.mark.parametrize(
    "shards", [[], [{"sequence_id": "s", "pairs": 11}], [{"sequence_id": "s", "pairs": 20}] * 2]
)
def test_invalid_or_insufficient_panel_rejected(shards):
    with pytest.raises(ValueError):
        temporal_schedule(shards, 10, 1, 0)


def test_temporal_config_records_exact_original_comparison_budget():
    path = Path("configs/experiment/registration_temporal_violation_cpu.yaml")
    cfg = read_config(path)
    assert cfg["updates_per_arm"] == 200
    assert cfg["updates_per_arm"] * cfg["batch_size"] == 160 * cfg["frames_per_sequence"]
    old = yaml.safe_load(
        Path("configs/experiment/registration_alignment_violation_cpu.yaml").read_text()
    )
    for key in (
        "seed",
        "initial_arm",
        "updates_per_arm",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "auxiliary_weights",
    ):
        assert cfg[key] == old[key]


def test_unmatched_budget_is_rejected(tmp_path):
    cfg = yaml.safe_load(
        Path("configs/experiment/registration_temporal_violation_cpu.yaml").read_text()
    )
    cfg["frames_per_sequence"] = 9
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="one pass"):
        read_config(path)
