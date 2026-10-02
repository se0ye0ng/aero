from collections import Counter

import pytest

from scripts.probe_registration_replay_preservation import replay_schedule, stack_samples


def test_uniform_deterministic_replay_with_identical_draws_and_no_batch_duplicates():
    identities = [str(i) for i in range(117)]
    batches = replay_schedule(identities)
    assert batches == replay_schedule(identities[::-1])
    assert batches != replay_schedule(identities, seed=1)
    assert len(batches) == 200
    assert all(len(b) == len(set(b)) == 4 for b in batches)
    counts = Counter(s for batch in batches for s in batch)
    assert set(counts) == set(identities)
    assert set(counts.values()) == {6, 7}


def test_invalid_replay_and_probe_inputs_rejected():
    with pytest.raises(ValueError, match="unique replay"):
        replay_schedule(["x"] * 4)
    with pytest.raises(ValueError, match="positive"):
        replay_schedule(["a", "b", "c", "d"], steps=0)
    with pytest.raises(ValueError, match="probe"):
        stack_samples([{"role": "probe"}])
